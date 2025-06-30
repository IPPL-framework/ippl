from pydantic import BaseModel, Field, Extra
from typing import Optional, List, Dict, Any

class Label(BaseModel):
    name: str
    value: str

class Attachment(BaseModel):
    name: str
    source: str
    type: Optional[str] = None

class StatusDetails(BaseModel):
    message: Optional[str] = None
    trace: Optional[str] = None

class TestResult(BaseModel, extra=Extra.allow):
    # Required core fields
    name: str
    status: str
    uuid: str
    testRunId: str
    historyId: str

    # Optional fields
    start: Optional[int] = None
    stop: Optional[int] = None
    duration: Optional[int] = None

    labels: Optional[List[Label]] = []
    parameters: Optional[List[Dict[str, Any]]] = []
    attachments: Optional[List[Attachment]] = []
    statusDetails: Optional[StatusDetails] = None
    executor: Optional[Dict[str, Any]] = None

class TestSuite(BaseModel):
    name: str
    tests: List[TestResult]
    labels: Optional[List[Label]] = []

class PipelineResult(BaseModel):
    pipeline_id: str
    suites: List[TestSuite]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

