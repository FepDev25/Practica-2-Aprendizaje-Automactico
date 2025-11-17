#!/usr/bin/env python3
"""
Script de prueba para verificar la integración del servicio LLM.
Ejecutar desde el directorio backend/
"""

import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

def test_llm_service():
    """Prueba básica del servicio LLM."""
    print("="*60)
    print("PRUEBA DE INTEGRACIÓN LLM - GEMINI")
    print("="*60)
    
    try:
        print("\n1. Importando servicio LLM...")
        from llm_service import get_llm_service
        
        print("✓ Módulo importado correctamente")
        
        print("\n2. Inicializando servicio...")
        llm_service = get_llm_service()
        print("✓ Servicio inicializado")
        
        print("\n3. Probando generación de mensaje individual...")
        mensaje = llm_service.generar_mensaje_prediccion(
            nombre_producto="Leche Entera 1L",
            sku="SKU-LECHE-001",
            fecha="2025-12-15",
            prediccion=15.5
        )
        
        print("\n--- MENSAJE GENERADO ---")
        print(mensaje)
        print("------------------------")
        
        print("\n4. Probando generación de mensaje múltiple...")
        mensaje_multiple = llm_service.generar_mensaje_multiple(
            fecha="2025-12-20",
            total_productos=45,
            predicciones_destacadas=[
                {"nombre": "Arroz Blanco 1kg", "prediccion": 8.5},
                {"nombre": "Aceite Vegetal 1L", "prediccion": 12.3},
                {"nombre": "Pan Integral 500g", "prediccion": 18.7},
                {"nombre": "Yogurt Natural 200ml", "prediccion": 156.0},
            ]
        )
        
        print("\n--- MENSAJE MÚLTIPLE GENERADO ---")
        print(mensaje_multiple)
        print("---------------------------------")
        
        print("\n" + "="*60)
        print("TODAS LAS PRUEBAS EXITOSAS")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}")
        print(f"Detalle: {e}")
        print("\n" + "="*60)
        print("VERIFICA:")
        print("  1. Archivo .env existe en backend/")
        print("  2. PROJECT_ID está configurado")
        print("  3. GOOGLE_APPLICATION_CREDENTIALS apunta al .json correcto")
        print("  4. El archivo .json de credenciales existe")
        print("  5. Tienes permisos en Vertex AI")
        print("="*60)
        return False


if __name__ == "__main__":
    success = test_llm_service()
    sys.exit(0 if success else 1)
