import os
import base64
from email.mime.text import MIMEText
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from app.core.database import google_tokens_collection

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]

COLLECTION_NAME = "google_user_tokens"


class GoogleWorkspaceService:

    @staticmethod
    async def save_credentials(user_id: str, creds_data: dict) -> None:
        """Guarda o actualiza las credenciales OAuth en MongoDB para un user_id."""
        creds_data["user_id"] = user_id
        creds_data["updated_at"] = datetime.now(timezone.utc)
        
        await google_tokens_collection.update_one(
            {"user_id": user_id},
            {"$set": creds_data},
            upsert=True
        )

    @staticmethod
    async def get_credentials(user_id: str) -> Optional[Credentials]:
        """Obtiene y refresca (si es necesario) las credenciales de un usuario."""
        token_doc = await google_tokens_collection.find_one({"user_id": user_id})
        if not token_doc:
            return None

        creds = Credentials(
            token=token_doc.get("token"),
            refresh_token=token_doc.get("refresh_token"),
            token_uri=token_doc.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
            scopes=token_doc.get("scopes", SCOPES),
        )

        # Si el access_token ha expirado pero tenemos refresh_token, lo refrescamos
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Actualizar el nuevo access_token en la base de datos
            await google_tokens_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "token": creds.token,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )

        return creds

    @staticmethod
    async def delete_credentials(user_id: str) -> bool:
        """Elimina la vinculación de la cuenta de Google."""
        result = await google_tokens_collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0

    @classmethod
    async def get_user_email(cls, user_id: str) -> Optional[str]:
        """Consulta el correo de la cuenta vinculada de Google."""
        creds = await cls.get_credentials(user_id)
        if not creds:
            return None
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        return profile.get("emailAddress")

    # ==========================================
    #             GOOGLE CALENDAR
    # ==========================================

    @classmethod
    async def list_upcoming_events(cls, user_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Devuelve los próximos eventos de la agenda."""
        creds = await cls.get_credentials(user_id)
        if not creds:
            raise ValueError("Cuenta de Google no conectada.")

        service = build("calendar", "v3", credentials=creds)
        now = datetime.now(timezone.utc).isoformat()

        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        formatted_events = []
        for ev in events:
            start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date"))
            end = ev.get("end", {}).get("dateTime", ev.get("end", {}).get("date"))
            formatted_events.append({
                "id": ev.get("id"),
                "summary": ev.get("summary", "Sin título"),
                "description": ev.get("description", ""),
                "start": start,
                "end": end,
                "htmlLink": ev.get("htmlLink")
            })
        return formatted_events

    @classmethod
    async def create_event(
        cls,
        user_id: str,
        summary: str,
        start_time_iso: str,
        end_time_iso: str,
        description: str = "",
        location: str = ""
    ) -> Dict[str, Any]:
        """Crea un nuevo evento en Google Calendar."""
        creds = await cls.get_credentials(user_id)
        if not creds:
            raise ValueError("Cuenta de Google no conectada.")

        service = build("calendar", "v3", credentials=creds)
        event_body = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start_time_iso},
            "end": {"dateTime": end_time_iso},
        }

        created_event = service.events().insert(calendarId="primary", body=event_body).execute()
        return {
            "id": created_event.get("id"),
            "summary": created_event.get("summary"),
            "htmlLink": created_event.get("htmlLink"),
            "start": created_event.get("start"),
            "end": created_event.get("end"),
        }

    # ==========================================
    #                  GMAIL
    # ==========================================

    @classmethod
    async def list_unread_emails(cls, user_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Recupera los últimos correos no leídos."""
        creds = await cls.get_credentials(user_id)
        if not creds:
            raise ValueError("Cuenta de Google no conectada.")

        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(
            userId="me",
            q="is:unread label:INBOX",
            maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        detailed_emails = []

        for msg in messages:
            msg_data = service.users().messages().get(
                userId="me",
                id=msg["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()

            headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
            detailed_emails.append({
                "id": msg["id"],
                "threadId": msg.get("threadId"),
                "snippet": msg_data.get("snippet"),
                "from": headers.get("From", "Desconocido"),
                "subject": headers.get("Subject", "Sin Asunto"),
                "date": headers.get("Date", "")
            })

        return detailed_emails

    @classmethod
    async def send_email(cls, user_id: str, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Envía un correo electrónico a través de Gmail."""
        creds = await cls.get_credentials(user_id)
        if not creds:
            raise ValueError("Cuenta de Google no conectada.")

        service = build("gmail", "v1", credentials=creds)
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw_message}
        ).execute()

        return {"id": sent.get("id"), "status": "sent"}

    @classmethod
    async def create_draft(cls, user_id: str, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Crea un borrador en Gmail sin enviarlo."""
        creds = await cls.get_credentials(user_id)
        if not creds:
            raise ValueError("Cuenta de Google no conectada.")

        service = build("gmail", "v1", credentials=creds)
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw_message}}
        ).execute()

        return {"id": draft.get("id"), "status": "draft_created"}