from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin, SourceMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.file import FileMetadata as PydanticFileMetadata

if TYPE_CHECKING:
    from letta.orm.files_agents import FileAgent
    from letta.orm.organization import Organization
    from letta.orm.passage import SourcePassage
    from letta.orm.source import Source


class FileMetadata(SqlalchemyBase, OrganizationMixin, SourceMixin):
    """Represents an uploaded file."""

    __tablename__ = "files"
    __pydantic_model__ = PydanticFileMetadata

    file_name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The name of the file.")
    file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The file path on the system.")
    file_type: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The type of the file.")
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, doc="The size of the file in bytes.")
    file_creation_date: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The creation date of the file.")
    file_last_modified_date: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The last modified date of the file.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="files", lazy="selectin")
    source: Mapped["Source"] = relationship("Source", back_populates="files", lazy="selectin")
    source_passages: Mapped[List["SourcePassage"]] = relationship(
        "SourcePassage", back_populates="file", lazy="selectin", cascade="all, delete-orphan"
    )
    file_agents: Mapped[List["FileAgent"]] = relationship("FileAgent", back_populates="file", lazy="selectin")
