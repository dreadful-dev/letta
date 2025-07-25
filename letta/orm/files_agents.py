import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.constants import CORE_MEMORY_SOURCE_CHAR_LIMIT, FILE_IS_TRUNCATED_WARNING
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.file import FileAgent as PydanticFileAgent

if TYPE_CHECKING:
    pass


class FileAgent(SqlalchemyBase, OrganizationMixin):
    """
    Join table between File and Agent.

    Tracks whether a file is currently "open" for the agent and
    the specific excerpt (grepped section) the agent is looking at.
    """

    __tablename__ = "files_agents"
    __table_args__ = (
        # (file_id, agent_id) must be unique
        UniqueConstraint("file_id", "agent_id", name="uq_file_agent"),
        # (file_name, agent_id) must be unique
        UniqueConstraint("agent_id", "file_name", name="uq_agent_filename"),
        # helpful indexes for look-ups
        Index("ix_file_agent", "file_id", "agent_id"),
        Index("ix_agent_filename", "agent_id", "file_name"),
    )
    __pydantic_model__ = PydanticFileAgent

    # single-column surrogate PK
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"file_agent-{uuid.uuid4()}",
    )

    # not part of the PK, but NOT NULL + FK
    file_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
        doc="ID of the file",
    )
    agent_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        doc="ID of the agent",
    )
    source_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        doc="ID of the source (denormalized from files.source_id)",
    )

    file_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Denormalized copy of files.file_name; unique per agent",
    )

    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, doc="True if the agent currently has the file open.")
    visible_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Portion of the file the agent is focused on.")
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="UTC timestamp when this agent last accessed the file.",
    )

    # relationships
    agent: Mapped["Agent"] = relationship(
        "Agent",
        back_populates="file_agents",
        lazy="selectin",
    )

    # TODO: This is temporary as we figure out if we want FileBlock as a first class citizen
    def to_pydantic_block(self) -> PydanticBlock:
        visible_content = self.visible_content if self.visible_content and self.is_open else ""

        # Truncate content and add warnings here when converting from FileAgent to Block
        if len(visible_content) > CORE_MEMORY_SOURCE_CHAR_LIMIT:
            truncated_warning = f"...[TRUNCATED]\n{FILE_IS_TRUNCATED_WARNING}"
            visible_content = visible_content[: CORE_MEMORY_SOURCE_CHAR_LIMIT - len(truncated_warning)]
            visible_content += truncated_warning

        return PydanticBlock(
            value=visible_content,
            label=self.file_name,  # use denormalized file_name instead of self.file.file_name
            read_only=True,
            metadata={"source_id": self.source_id},  # use denormalized source_id
            limit=CORE_MEMORY_SOURCE_CHAR_LIMIT,
        )
