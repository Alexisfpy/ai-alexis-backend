import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow

from app.services.google_workspace_service import GoogleWorkspaceService, SCOPES
from app.core.database import google_tokens_collection

router = APIRouter(prefix="/api/v1/google", tags=["Google Workspace Auth"])


def get_oauth_flow(state: str = None) -> Flow:
    """Construye el flujo OAuth utilizando las variables de entorno."""
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/v1/google/callback")
    client_config = {
        "web": {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri]
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        state=state,
        redirect_uri=redirect_uri
    )


@router.get("/login")
def google_login(user_id: str = Query(..., description="ID de usuario de Clerk")):
    """
    Inicia el flujo OAuth.
    Redirige a Google pasando el user_id en el parámetro 'state'.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500, 
            detail="Credenciales de Google OAuth no configuradas en las variables de entorno."
        )

    flow = get_oauth_flow(state=user_id)
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )

    # Persistir temporalmente el code_verifier de PKCE generado por Flow
    if getattr(flow, "code_verifier", None):
        google_tokens_collection.update_one(
            {"user_id": user_id},
            {"$set": {"code_verifier": flow.code_verifier}},
            upsert=True
        )

    return RedirectResponse(url=authorization_url)


@router.get("/callback")
def google_callback(code: str = Query(...), state: str = Query(...)):
    """
    Recibe el código de Google tras la autorización,
    recupera el code_verifier, intercambia los tokens y los guarda en MongoDB.
    """
    user_id = state
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

    try:
        flow = get_oauth_flow(state=state)

        # Restaurar el code_verifier previo para que el intercambio no falle
        token_doc = google_tokens_collection.find_one({"user_id": user_id})
        if token_doc and token_doc.get("code_verifier"):
            flow.code_verifier = token_doc["code_verifier"]

        flow.fetch_token(code=code)
        creds = flow.credentials

        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        }

        # Guardar credenciales definitivas y limpiar el code_verifier temporal
        GoogleWorkspaceService.save_credentials(user_id, creds_data)
        google_tokens_collection.update_one(
            {"user_id": user_id},
            {"$unset": {"code_verifier": ""}}
        )

        return RedirectResponse(url=f"{frontend_url}/?google_auth=success")

    except Exception as e:
        return RedirectResponse(url=f"{frontend_url}/?google_auth=error&detail={str(e)}")


@router.get("/status/{user_id}")
def get_connection_status(user_id: str):
    """Devuelve si el usuario tiene una cuenta vinculada y el email asociado."""
    try:
        email = GoogleWorkspaceService.get_user_email(user_id)
        if email:
            return {"connected": True, "email": email}
        return {"connected": False, "email": None}
    except Exception:
        return {"connected": False, "email": None}


@router.post("/disconnect/{user_id}")
def disconnect_google(user_id: str):
    """Elimina las credenciales del usuario de la base de datos."""
    success = GoogleWorkspaceService.delete_credentials(user_id)
    return {"success": success, "message": "Cuenta desconectada correctamente."}