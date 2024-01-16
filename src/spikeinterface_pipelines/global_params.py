from pydantic import BaseModel, Field
from typing import Union


class JobKwargs(BaseModel):
    n_jobs: Union[int, float] = Field(default=-1, description="The number of jobs to run in parallel.")
    chunk_duration: str = Field(default="1s", description="The duration of the chunks to process.")
    progress_bar: bool = Field(default=False, description="Whether to display a progress bar.")
