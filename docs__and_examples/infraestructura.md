# Guía de Infraestructura y Despliegue

- Infraestructura del sistema, organización de máquinas virtuales en Google Cloud, archivos Docker necesarios para cada capa y archivo docker-compose para orquestación.

## 1. Arquitectura en Google Cloud (GCC)

- Se desplegarán tres máquinas virtuales independientes en Google Cloud Compute Engine:

- VM 1 — Frontend (Angular + Nginx)
  - Expone puerto 80/443
  - Corre contenedor de Nginx que sirve el build de Angular

- VM 2 — Backend (FastAPI + Modelo ML)
  - Expone puerto 8000 (solo accesible desde VM frontend)
  - Corre API FastAPI + modelo de Machine Learning en Docker

- VM 3 — Base de Datos (PostgreSQL)
  - Expone puerto 5432 (solo accesible desde VM backend)
  - Contenedor Docker con PostgreSQL

## 2. Estructura del Proyecto

project/
    frontend/
        Dockerfile
    backend/
        Dockerfile
    database/
        docker-compose.yml (opcional para VM de base de datos)
    docker-compose.yml (para backend y frontend)

## 3. Reglas de Firewall

- Frontend:
  - Permitir 80/443 desde Internet

- Backend:
  - Permitir 8000 solo desde VM frontend

- Base de Datos:
  - Permitir 5432 solo desde backend
