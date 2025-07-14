from dict_hash import Hashable, sha256
from matchms import Spectrum as MatchmsSpectrum


class Spectrum(MatchmsSpectrum, Hashable):
    """A Spectrum class that extends the matchms Spectrum so that we can use it in the cache_decorator package."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a consistent hash of the Spectrum object."""
        return sha256(
            {
                "spectrum_hash": super().spectrum_hash(),
                "metadata_hash": super().metadata_hash(),
            },
            use_approximation=use_approximation,
        )
