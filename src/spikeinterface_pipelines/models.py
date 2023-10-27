from pydantic import BaseModel, Field


class JobKwargs(BaseModel):
    n_jobs: int = Field(-1, description="The number of jobs to run in parallel.")
    chunk_duration: str = Field("1s", description="The duration of the chunks to process.")
    progress_bar: bool = Field(True, description="Whether to display a progress bar.")