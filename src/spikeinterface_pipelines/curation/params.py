from pydantic import BaseModel, Field


class CurationParams(BaseModel):
    """
    Curation parameters.
    """
    curation_query: str = Field(
        default="isi_violations_ratio < 0.5 and amplitude_cutoff < 0.1 and presence_ratio > 0.8",
        description=(
            "Query to select units to keep after curation. "
            "Default is 'isi_violations_ratio < 0.5 and amplitude_cutoff < 0.1 and presence_ratio > 0.8'."
        )
    )