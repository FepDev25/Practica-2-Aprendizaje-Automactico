"""
Esquemas Pydantic para validación de datos
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import List, Optional
from datetime import datetime

# ==================== SCHEMAS DE ROLES ====================

class RoleBase(BaseModel):
    """Schema base para roles"""
    name: str = Field(..., min_length=1, max_length=50, description="Nombre del rol")
    description: Optional[str] = Field(None, max_length=255, description="Descripción del rol")

class RoleCreate(RoleBase):
    """Schema para crear un rol"""
    pass

class RoleResponse(RoleBase):
    """Schema para la respuesta de un rol"""
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# ==================== SCHEMAS DE USUARIOS ====================

class UserBase(BaseModel):
    """Schema base para usuarios"""
    username: str = Field(..., min_length=3, max_length=50, description="Nombre de usuario único")
    email: EmailStr = Field(..., description="Email del usuario")
    enabled: bool = Field(default=True, description="Estado del usuario")

class UserCreate(UserBase):
    """Schema para crear un usuario"""
    password: str = Field(..., min_length=6, max_length=255, description="Contraseña del usuario")

class UserUpdate(BaseModel):
    """Schema para actualizar un usuario (todos los campos opcionales)"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6, max_length=255)
    enabled: Optional[bool] = None

class UserResponse(UserBase):
    """Schema para la respuesta de un usuario"""
    id: int
    created_at: datetime
    updated_at: datetime
    roles: List[RoleResponse] = []

    model_config = ConfigDict(from_attributes=True)

# ==================== SCHEMAS DE USER_ROLES ====================

class UserRoleAssignment(BaseModel):
    """Schema para asignar un rol a un usuario"""
    user_id: int
    role_id: int
