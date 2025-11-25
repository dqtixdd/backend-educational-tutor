# backend/memory_sqlalchemy.py
import json
from datetime import datetime
from passlib.context import CryptContext
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError

from db import SessionLocal
from models import User, Conversation, Message

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class MemoryStore:
    def __init__(self):
        pass  # no manual _init_db needed, tables created via Base.metadata

    def _session(self):
        return SessionLocal()

    # --- USER MANAGEMENT ---
    def create_user(self, email, username, password):
        password_hash = pwd_context.hash(password)
        with self._session() as db:
            user = User(email=email, username=username, password_hash=password_hash)
            db.add(user)
            try:
                db.commit()
                return True
            except IntegrityError:
                db.rollback()
                return False

    def verify_user(self, email, password):
        with self._session() as db:
            user = db.get(User, email)
            if user and pwd_context.verify(password, user.password_hash):
                return {"email": user.email, "username": user.username}
            return None

    # --- CONVERSATIONS ---
    def create_conversation(self, cid, email, title="New Chat"):
        with self._session() as db:
            conv = db.get(Conversation, cid)
            if not conv:
                conv = Conversation(
                    id=cid,
                    title=title,
                    user_email=email,
                    updated_at=datetime.utcnow(),
                )
                db.add(conv)
                db.commit()

    def get_all_conversations(self, email):
        with self._session() as db:
            stmt = (
                select(Conversation)
                .where(Conversation.user_email == email)
                .order_by(Conversation.updated_at.desc())
            )
            return [
                {"id": c.id, "title": c.title}
                for c in db.scalars(stmt).all()
            ]

    def delete_conversation(self, cid, email):
        with self._session() as db:
            stmt = select(Conversation).where(
                Conversation.id == cid,
                Conversation.user_email == email,
            )
            conv = db.scalars(stmt).first()
            if conv:
                db.delete(conv)  # cascade deletes messages
                db.commit()

    # --- MESSAGES ---
    def add_message(self, cid, role, content, sources=None):
        with self._session() as db:
            msg = Message(
                conversation_id=cid,
                role=role,
                content=content,
                sources=json.dumps(sources) if sources else None,
                created_at=datetime.utcnow(),
            )
            db.add(msg)
            conv = db.get(Conversation, cid)
            if conv:
                conv.updated_at = datetime.utcnow()
            db.commit()

    def get_messages(self, cid):
        with self._session() as db:
            stmt = (
                select(Message)
                .where(Message.conversation_id == cid)
                .order_by(Message.id.asc())
            )
            msgs = db.scalars(stmt).all()
            return [
                {
                    "role": m.role,
                    "text": m.content,
                    "sources": json.loads(m.sources) if m.sources else None,
                }
                for m in msgs
            ]

    def transcript_text(self, cid, last_n=8):
        with self._session() as db:
            stmt = (
                select(Message)
                .where(Message.conversation_id == cid)
                .order_by(Message.id.desc())
                .limit(last_n)
            )
            rows = list(reversed(db.scalars(stmt).all()))
            return "\n".join(
                ["User: " + m.content if m.role == "user" else "Assistant: " + m.content
                 for m in rows]
            )
