from typing import List, Optional, Union

from letta.orm.provider import Provider as ProviderModel
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.providers import Provider as PydanticProvider
from letta.schemas.providers import ProviderCheck, ProviderCreate, ProviderUpdate
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.utils import enforce_types


class ProviderManager:

    @enforce_types
    def create_provider(self, request: ProviderCreate, actor: PydanticUser) -> PydanticProvider:
        """Create a new provider if it doesn't already exist."""
        with db_registry.session() as session:
            provider_create_args = {**request.model_dump(), "provider_category": ProviderCategory.byok}
            provider = PydanticProvider(**provider_create_args)

            if provider.name == provider.provider_type.value:
                raise ValueError("Provider name must be unique and different from provider type")

            # Assign the organization id based on the actor
            provider.organization_id = actor.organization_id

            # Lazily create the provider id prior to persistence
            provider.resolve_identifier()

            new_provider = ProviderModel(**provider.model_dump(to_orm=True, exclude_unset=True))
            new_provider.create(session, actor=actor)
            return new_provider.to_pydantic()

    @enforce_types
    def update_provider(self, provider_id: str, provider_update: ProviderUpdate, actor: PydanticUser) -> PydanticProvider:
        """Update provider details."""
        with db_registry.session() as session:
            # Retrieve the existing provider by ID
            existing_provider = ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)

            # Update only the fields that are provided in ProviderUpdate
            update_data = provider_update.model_dump(to_orm=True, exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_provider, key, value)

            # Commit the updated provider
            existing_provider.update(session, actor=actor)
            return existing_provider.to_pydantic()

    @enforce_types
    def delete_provider_by_id(self, provider_id: str, actor: PydanticUser):
        """Delete a provider."""
        with db_registry.session() as session:
            # Clear api key field
            existing_provider = ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)
            existing_provider.api_key = None
            existing_provider.update(session, actor=actor)

            # Soft delete in provider table
            existing_provider.delete(session, actor=actor)

            session.commit()

    @enforce_types
    def list_providers(
        self,
        actor: PydanticUser,
        name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        after: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[PydanticProvider]:
        """List all providers with optional pagination."""
        filter_kwargs = {}
        if name:
            filter_kwargs["name"] = name
        if provider_type:
            filter_kwargs["provider_type"] = provider_type
        with db_registry.session() as session:
            providers = ProviderModel.list(
                db_session=session,
                after=after,
                limit=limit,
                actor=actor,
                **filter_kwargs,
            )
            return [provider.to_pydantic() for provider in providers]

    @enforce_types
    def get_provider_id_from_name(self, provider_name: Union[str, None], actor: PydanticUser) -> Optional[str]:
        providers = self.list_providers(name=provider_name, actor=actor)
        return providers[0].id if providers else None

    @enforce_types
    def get_override_key(self, provider_name: Union[str, None], actor: PydanticUser) -> Optional[str]:
        providers = self.list_providers(name=provider_name, actor=actor)
        return providers[0].api_key if providers else None

    @enforce_types
    def check_provider_api_key(self, provider_check: ProviderCheck) -> None:
        provider = PydanticProvider(
            name=provider_check.provider_type.value,
            provider_type=provider_check.provider_type,
            api_key=provider_check.api_key,
            provider_category=ProviderCategory.byok,
        ).cast_to_subtype()

        # TODO: add more string sanity checks here before we hit actual endpoints
        if not provider.api_key:
            raise ValueError("API key is required")

        provider.check_api_key()
