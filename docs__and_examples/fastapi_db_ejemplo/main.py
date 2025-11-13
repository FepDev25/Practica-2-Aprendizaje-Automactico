"""
FastAPI CRUD para sistema de usuarios con PostgreSQL
Base de datos: ejemplo1
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import models
import schemas
import crud
from database import engine, get_db

# Crear las tablas (solo si no existen, pero en tu caso ya existen)
# models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sistema de Usuarios API",
    description="CRUD completo para gestión de usuarios, roles y relaciones",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENDPOINTS DE USUARIOS ====================

@app.get("/", tags=["Root"])
def read_root():
    """Endpoint de bienvenida"""
    return {
        "message": "API de Gestión de Usuarios",
        "version": "1.0.0",
        "endpoints": {
            "users": "/users",
            "roles": "/roles",
            "docs": "/docs"
        }
    }

@app.get("/users/", response_model=List[schemas.UserResponse], tags=["Users"])
def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Obtener todos los usuarios"""
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=schemas.UserResponse, tags=["Users"])
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Obtener un usuario específico por ID"""
    user = crud.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

@app.post("/users/", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Crear un nuevo usuario"""
    # Verificar si el username ya existe
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="El username ya está registrado")
    
    # Verificar si el email ya existe
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    return crud.create_user(db=db, user=user)

@app.put("/users/{user_id}", response_model=schemas.UserResponse, tags=["Users"])
def update_existing_user(user_id: int, user: schemas.UserUpdate, db: Session = Depends(get_db)):
    """Actualizar un usuario existente"""
    db_user = crud.update_user(db, user_id=user_id, user=user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return db_user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Users"])
def delete_existing_user(user_id: int, db: Session = Depends(get_db)):
    """Eliminar un usuario"""
    success = crud.delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return None

@app.patch("/users/{user_id}/enable", response_model=schemas.UserResponse, tags=["Users"])
def enable_user(user_id: int, db: Session = Depends(get_db)):
    """Habilitar un usuario"""
    db_user = crud.toggle_user_status(db, user_id=user_id, enabled=True)
    if db_user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return db_user

@app.patch("/users/{user_id}/disable", response_model=schemas.UserResponse, tags=["Users"])
def disable_user(user_id: int, db: Session = Depends(get_db)):
    """Deshabilitar un usuario"""
    db_user = crud.toggle_user_status(db, user_id=user_id, enabled=False)
    if db_user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return db_user

# ==================== ENDPOINTS DE ROLES ====================

@app.get("/roles/", response_model=List[schemas.RoleResponse], tags=["Roles"])
def get_all_roles(db: Session = Depends(get_db)):
    """Obtener todos los roles"""
    roles = crud.get_roles(db)
    return roles

@app.get("/roles/{role_id}", response_model=schemas.RoleResponse, tags=["Roles"])
def get_role(role_id: int, db: Session = Depends(get_db)):
    """Obtener un rol específico por ID"""
    role = crud.get_role(db, role_id=role_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Rol no encontrado")
    return role

@app.post("/roles/", response_model=schemas.RoleResponse, status_code=status.HTTP_201_CREATED, tags=["Roles"])
def create_new_role(role: schemas.RoleCreate, db: Session = Depends(get_db)):
    """Crear un nuevo rol"""
    db_role = crud.get_role_by_name(db, name=role.name)
    if db_role:
        raise HTTPException(status_code=400, detail="El nombre del rol ya existe")
    return crud.create_role(db=db, role=role)

# ==================== ENDPOINTS DE USER_ROLES ====================

@app.post("/users/{user_id}/roles/{role_id}", response_model=schemas.UserResponse, tags=["User-Roles"])
def assign_role_to_user(user_id: int, role_id: int, db: Session = Depends(get_db)):
    """Asignar un rol a un usuario"""
    result = crud.add_role_to_user(db, user_id=user_id, role_id=role_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Usuario o rol no encontrado")
    return result

@app.delete("/users/{user_id}/roles/{role_id}", response_model=schemas.UserResponse, tags=["User-Roles"])
def remove_role_from_user(user_id: int, role_id: int, db: Session = Depends(get_db)):
    """Remover un rol de un usuario"""
    result = crud.remove_role_from_user(db, user_id=user_id, role_id=role_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Usuario o rol no encontrado")
    return result

@app.get("/users/{user_id}/roles", response_model=List[schemas.RoleResponse], tags=["User-Roles"])
def get_user_roles(user_id: int, db: Session = Depends(get_db)):
    """Obtener todos los roles de un usuario"""
    user = crud.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user.roles

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
