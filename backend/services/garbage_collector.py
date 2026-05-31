import asyncio
import logging

log = logging.getLogger("qwen2api.gc")


async def _delete_stale_chat(client, acc, chat_id: str, chat_id_pool) -> None:
    if chat_id_pool is not None and await chat_id_pool.contains(acc.email, chat_id):
        log.debug("[GC] skip pooled chat account=%s chat_id=%s", acc.email, chat_id)
        return
    if chat_id_pool is not None:
        await chat_id_pool.invalidate(acc.email, chat_id)
    try:
        await client.delete_chat(acc.token, chat_id)
        log.debug("[GC] deleted stale chat account=%s chat_id=%s", acc.email, chat_id)
    except Exception as exc:
        log.debug("[GC] delete stale chat failed account=%s chat_id=%s error=%s", acc.email, chat_id, exc)


async def garbage_collect_chats(app):
    """
    Every 15 minutes, delete stale API-created chats.
    API-created chats are identified by title prefix api_.
    Active session-affinity chat IDs are kept.
    """
    client = app.state.qwen_client
    while True:
        await asyncio.sleep(900)  # 15 minutes
        log.info("[GC] scanning stale API chats...")
        pool = client.account_pool
        active_chat_ids = app.state.session_affinity.active_chat_ids()
        executor = getattr(client, "executor", None)
        if executor is not None and hasattr(executor, "active_chat_ids"):
            active_chat_ids = set(active_chat_ids) | executor.active_chat_ids()
        chat_id_pool = getattr(app.state, "chat_id_pool", None) or getattr(executor, "chat_id_pool", None)
        for acc in pool.accounts:
            if not acc.is_available():
                continue
            try:
                pooled_chat_ids = await chat_id_pool.chat_ids(acc.email) if chat_id_pool is not None else set()
                chats = await client.list_chats(acc.token, limit=50)
                for c in chats:
                    if not isinstance(c, dict):
                        continue
                    chat_id = str(c.get("id") or "")
                    if not c.get("title", "").startswith("api_"):
                        continue
                    if chat_id and chat_id in active_chat_ids:
                        continue
                    if chat_id and chat_id in pooled_chat_ids:
                        continue
                    asyncio.create_task(_delete_stale_chat(client, acc, chat_id, chat_id_pool))
            except Exception as e:
                log.warning(f"[GC] account {acc.email} cleanup failed: {e}")
