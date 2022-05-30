from __future__ import annotations

from functools import partial
from math import floor
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import vapoursynth as vs
from pyfftw import FFTW, empty_aligned  # type: ignore

from .utils import (
    cuda_available, cuda_error, cuda_stream, cufft, cupy, cupyx, fftw_cpu_kwargs, is_cuda_101
)

core = vs.core

__all__ = ['FFTSpectrum']


_scale_val = 297.8690509933263


def _fast_roll(fdst: Any, fsrc: Any, yh: int, xh: int) -> None:
    fdst[:-yh, :-xh] = fsrc[yh:, xh:]
    fdst[:-yh, -xh:] = fsrc[yh:, :xh]
    fdst[-yh:, :-xh] = fsrc[:yh, xh:]
    fdst[-yh:, -xh:] = fsrc[:yh, :xh]


def _fftspectrum_cpu_modifyframe(
    f: List[vs.VideoFrame], n: int, rollfunc: Any, fftw_obj: Any
) -> vs.VideoFrame:
    fdst = f[1].copy()

    farr: np.typing.NDArray[np.complex64] = np.asarray(f[0][0])
    farr = np.ascontiguousarray(farr)
    farr = farr.astype(np.complex64)

    fftw_obj(farr, fftw_obj.output_array)

    fdst_arr: np.typing.NDArray[np.float32] = np.asarray(fdst[0])

    rollfunc(fdst_arr, fftw_obj.output_array.real)

    return fdst


if cuda_available:
    def _fftspectrum_gpu_modifyframe(
        f: List[vs.VideoFrame], n: int, threshold: float,
        rollfunc: Any, cuda_plan: Any, fout: cupy.typing.NDArray[cupy.complex64]
    ) -> vs.VideoFrame:
        fdst = f[1].copy()

        farr = cupy.asarray(f[0][0])
        farr = cupy.ascontiguousarray(farr)
        farr = farr.astype(cupy.complex64)

        cuda_plan.fft(farr, fout, cufft.CUFFT_FORWARD)

        fft_norm = cupy.log(cupy.abs(fout.real))

        maxval = fft_norm[0][0]

        fft_norm = cupy.where(
            fft_norm > (maxval / threshold), fft_norm * (_scale_val / maxval), 0
        ).clip(0, 255)

        rollfunc(fout.real, fft_norm)

        fout.real.astype(cupy.uint8).get(cuda_stream, 'C', np.asarray(fdst[0]))

        return fdst

_fft_modifyframe_cache: Dict[Tuple[Tuple[int, int], bool], Callable[[vs.VideoFrame, int], vs.VideoFrame]] = {}


def FFTSpectrum(
    clip: vs.VideoNode, threshold: float = 2.25, target_size: Tuple[int, int] | None = None, cuda: bool | None = None
) -> vs.VideoNode:
    assert clip.format

    cuda = cuda_available if cuda is None else bool(cuda)

    if cuda and not cuda_available:
        raise ValueError(
            f"FFTSpectrum: Cuda acceleration isn't available!\nError: `{cuda_error}`"
        )

    if clip.format.bits_per_sample != 8 or clip.format.sample_type != vs.INTEGER:
        clip = clip.resize.Bicubic(
            range=None, range_in=None,
            dither_type='error_diffusion',
            format=clip.format.replace(sample_type=vs.INTEGER, bits_per_sample=8).id
        )

    shape = (clip.height, clip.width)

    cache_key = (shape, cuda)

    if cache_key not in _fft_modifyframe_cache:
        if cuda:
            fft_cuplan_zeros = cupy.empty(shape, cupy.complex64, 'C')

            cuda_plan = cupyx.scipy.fftpack.get_fft_plan(
                fft_cuplan_zeros, shape, fftw_cpu_kwargs['axes']
            )
        else:
            fft_src_aligned_zeros = empty_aligned(shape, np.complex64)
            fft_out_aligned_zeros = empty_aligned(shape, np.complex64)
            fftw_obj = FFTW(
                fft_src_aligned_zeros, fft_out_aligned_zeros, **fftw_cpu_kwargs
            )

        rollfunc = partial(_fast_roll, xh=clip.width // 2, yh=clip.height // 2)

        if cuda:
            _fft_modifyframe_cache[cache_key] = partial(
                _fftspectrum_gpu_modifyframe,
                rollfunc=rollfunc, cuda_plan=cuda_plan, fout=fft_cuplan_zeros
            )
        else:
            _fft_modifyframe_cache[cache_key] = partial(
                _fftspectrum_cpu_modifyframe, rollfunc=rollfunc, fftw_obj=fftw_obj
            )

    _modify_frame_func = _fft_modifyframe_cache[cache_key]

    if cuda:
        _modify_frame_func = partial(_modify_frame_func, threshold=threshold)

    blankclip = clip.std.BlankClip(
        format=vs.GRAY8 if cuda else vs.GRAYS, color=0, keep=True
    )

    fftclip = blankclip.std.ModifyFrame([clip, blankclip], _modify_frame_func)

    if target_size:
        max_width, max_height = target_size

        if clip.width != max_width or clip.height != max_height:
            w_diff, h_diff = max_width - clip.width, max_height - clip.height
            w_pad, w_mod = (floor(w_diff / 2), w_diff % 2) if w_diff > 0 else (0, 0)
            h_pad, h_mod = (floor(h_diff / 2), h_diff % 2) if h_diff > 0 else (0, 0)

            fftclip = fftclip.std.AddBorders(w_pad, w_pad + w_mod, h_pad, h_pad + h_mod)

            if w_mod or h_mod:
                fftclip = fftclip.resize.Bicubic(src_top=h_mod / 2, src_left=w_mod / 2)

    if not cuda:
        fftscaled = fftclip.akarin.Expr('x abs log 10 /')
        fftstats = fftscaled.std.PlaneStats(prop='P')
        fftclip = fftstats.akarin.Expr(
            f'x x.PMax {threshold} / > x {_scale_val} x.PMax / * 0 ?', vs.GRAY8
        )

    return fftclip
