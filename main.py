import os
import asyncio
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY.strip()) if GROQ_API_KEY else None

WORKING_MODELS: List[str] = []

def discover_working_models(target_count: int = 5) -> List[str]:
    """Groq API থেকে মডেল সংগ্রহ ও যাচাই (একবারই চলবে স্টার্টআপে)"""
    global WORKING_MODELS
    if len(WORKING_MODELS) >= target_count:
        return WORKING_MODELS

    if not client:
        print("[ERROR] GROQ_API_KEY পাওয়া যায়নি!")
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    blocked_keywords = [
        "whisper", "tts", "audio", "embed", "guard", 
        "classifier", "moderation", "safeguard", "vision",
        "qwen", "r1", "preview"
    ]

    try:
        models_data = client.models.list().data
        available_ids = [
            m.id for m in models_data 
            if getattr(m, "active", True) and not any(blocked in m.id.lower() for blocked in blocked_keywords)
        ]
    except Exception as e:
        print(f"[WARN] মডেল তালিকা আনতে ব্যর্থ: {e}")
        available_ids = []

    verified_models = []
    test_messages = [
        {"role": "system", "content": "You are a test helper."},
        {"role": "user", "content": "hi"}
    ]

    for model_id in available_ids:
        try:
            res = client.chat.completions.create(
                messages=test_messages,
                model=model_id,
                max_tokens=10
            )
            if res and res.choices:
                verified_models.append(model_id)
                print(f"[MODEL READY] {model_id}")
                if len(verified_models) == target_count:
                    break
        except Exception:
            continue

    if not verified_models:
        verified_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    while len(verified_models) < target_count:
        verified_models.append(verified_models[len(verified_models) % len(verified_models)])

    WORKING_MODELS = verified_models[:target_count]
    print(f"[FINAL MODELS LOCKED] {WORKING_MODELS}")
    return WORKING_MODELS

# সার্ভার স্টার্ট হওয়ার সাথে সাথে ব্যাকগ্রাউন্ডে মডেল রেডি করা
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, discover_working_models)
    yield

app = FastAPI(lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    prompt: str
    history: List[Message] = []

@app.get("/ping")
def ping():
    return {
        "status": "alive", 
        "assigned_models": WORKING_MODELS if len(WORKING_MODELS) == 5 else "Loading..."
    }

async def run_agent(role_instruction: str, context: str, user_input: str, model_name: str) -> str:
    loop = asyncio.get_event_loop()
    
    messages = [
        {"role": "system", "content": f"{role_instruction}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": user_input}
    ]

    def call_groq():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower():
                # ফলব্যাক মডেল
                fallback = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                return fallback.choices[0].message.content
            raise e

    return await loop.run_in_executor(None, call_groq)

@app.post("/api/chat")
async def chat_multi_agent(payload: ChatPayload):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="প্রম্পট খালি রাখা যাবে না")

    models = discover_working_models()
    history_context = "\n".join([f"{m.role.upper()}: {m.content}" for m in payload.history[-4:]])

    try:
        a1_plan = await run_agent(
            "You are Agent 1 (Orchestrator). Break down the user problem into clear technical sub-goals concisely.",
            history_context, payload.prompt, models[0]
        )

        a2_build = await run_agent(
            "You are Agent 2 (Builder/Coder). Generate the core structural logic or concise solution based on Agent 1's plan.",
            history_context, a1_plan, models[1]
        )

        a3_critic = await run_agent(
            "You are Agent 3 (Critic/QA). Spot bugs, edge cases, security issues, or flaws in Agent 2's output concisely.",
            history_context, a2_build, models[2]
        )

        a4_refine = await run_agent(
            "You are Agent 4 (Optimizer). Fix issues spotted by Agent 3 and polish the deliverable.",
            history_context, f"Draft:\n{a2_build}\n\nCritique:\n{a3_critic}", models[3]
        )

        a5_final = await run_agent(
            "You are Agent 5 (Presenter). Prepare the final response for the user. Deliver an impeccably organized, crystal-clear deliverable without meta-commentary.",
            history_context, a4_refine, models[4]
        )

        return {
            "models_used": {
                "agent_1": models[0],
                "agent_2": models[1],
                "agent_3": models[2],
                "agent_4": models[3],
                "agent_5": models[4],
            },
            "agent_1": a1_plan,
            "agent_2": a2_build,
            "agent_3": a3_critic,
            "agent_4": a4_refine,
            "final_output": a5_final
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Autonomous 5-Agent Workspace</title>
        <style>
            :root {
                --bg: #0d1117;
                --panel: #161b22;
                --border: #30363d;
                --accent: #238636;
                --text: #c9d1d9;
                --highlight: #58a6ff;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background-color: var(--bg); color: var(--text); padding: 20px; display: flex; justify-content: center; }
            .container { max-width: 950px; width: 100%; display: flex; flex-direction: column; gap: 20px; }
            header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; }
            header h1 { font-size: 1.4rem; color: var(--highlight); }
            .clear-btn { background: #21262d; border: 1px solid var(--border); color: #f85149; padding: 6px 12px; border-radius: 6px; cursor: pointer; }
            
            #chatLog { display: flex; flex-direction: column; gap: 20px; }
            .turn { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }
            .user-msg { font-size: 1rem; font-weight: 600; color: #f0f6fc; margin-bottom: 15px; border-left: 3px solid var(--highlight); padding-left: 10px; }
            
            .final-card { background: #0d1117; border: 1px solid #238636; border-radius: 8px; padding: 18px; margin-bottom: 15px; }
            .final-title { font-size: 1rem; font-weight: 700; color: #7ee787; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
            .model-badge { font-size: 0.75rem; background: #21262d; padding: 2px 8px; border-radius: 10px; color: #8b949e; font-weight: normal; }
            .output-text { white-space: pre-wrap; word-break: break-word; font-size: 0.95rem; line-height: 1.6; color: #f0f6fc; }

            .agent-tray { display: flex; flex-direction: column; gap: 8px; }
            details { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
            summary { padding: 10px 14px; cursor: pointer; font-size: 0.85rem; font-weight: 600; color: #8b949e; user-select: none; }
            summary:hover { background: #1f242c; color: #c9d1d9; }
            .agent-content { padding: 12px; font-size: 0.85rem; white-space: pre-wrap; word-break: break-word; color: #8b949e; border-top: 1px solid var(--border); background: #090d12; font-family: monospace; }
            
            .input-box { position: sticky; bottom: 20px; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
            textarea { width: 100%; height: 75px; background: #0d1117; border: 1px solid var(--border); border-radius: 6px; color: #f0f6fc; padding: 10px; resize: none; font-size: 0.95rem; outline: none; }
            textarea:focus { border-color: var(--highlight); }
            button.submit { background: var(--accent); color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; align-self: flex-end; }
            button.submit:disabled { opacity: 0.5; cursor: not-allowed; }
            .status { font-size: 0.85rem; color: #8b949e; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Autonomous 5-Agent Workspace</h1>
                <button class="clear-btn" onclick="clearMemory()">Clear Memory</button>
            </header>

            <div id="chatLog"></div>

            <div class="input-box">
                <textarea id="promptInput" placeholder="আপনার নির্দেশ বা সমস্যা লিখুন..."></textarea>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span id="status" class="status">প্রস্তুত</span>
                    <button id="submitBtn" class="submit" onclick="executePrompt()">Execute</button>
                </div>
            </div>
        </div>

        <script>
            let chatHistory = JSON.parse(localStorage.getItem('agent_history') || '[]');

            function clearMemory() {
                chatHistory = [];
                localStorage.removeItem('agent_history');
                document.getElementById('chatLog').innerHTML = '';
            }

            async function executePrompt() {
                const input = document.getElementById('promptInput');
                const prompt = input.value.trim();
                if (!prompt) return;

                const submitBtn = document.getElementById('submitBtn');
                const status = document.getElementById('status');
                submitBtn.disabled = true;
                status.innerText = "৫টি এজেন্ট ক্রমান্বয়ে সমাধান প্রস্তুত করছে...";

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: prompt, history: chatHistory })
                    });
                    
                    const textData = await res.text();
                    let data;
                    try {
                        data = JSON.parse(textData);
                    } catch (err) {
                        throw new Error("সার্ভার থেকে সঠিক রেসপন্স আসেনি (সম্ভবত টাইমআউট বা ইন্টারনাল এরর)");
                    }

                    if (!res.ok) throw new Error(data.detail || "Execution failed");

                    chatHistory.push({ role: "user", content: prompt });
                    chatHistory.push({ role: "assistant", content: data.final_output });
                    localStorage.setItem('agent_history', JSON.stringify(chatHistory));

                    renderTurn(prompt, data);
                    input.value = '';
                    status.innerText = "সম্পন্ন";
                } catch (e) {
                    status.innerText = "ত্রুটি: " + e.message;
                } finally {
                    submitBtn.disabled = false;
                }
            }

            function renderTurn(prompt, data) {
                const log = document.getElementById('chatLog');
                const turnDiv = document.createElement('div');
                turnDiv.className = 'turn';

                const m = data.models_used || {};

                turnDiv.innerHTML = `
                    <div class="user-msg">${escapeHtml(prompt)}</div>
                    
                    <div class="final-card">
                        <div class="final-title">
                            <span>✓ Agent 5: Verified & Structured Output</span>
                            <span class="model-badge">${escapeHtml(m.agent_5 || 'Active')}</span>
                        </div>
                        <div class="output-text">${escapeHtml(data.final_output)}</div>
                    </div>

                    <div class="agent-tray">
                        <details>
                            <summary>Agent 1: Orchestrator [${escapeHtml(m.agent_1 || '')}]</summary>
                            <div class="agent-content">${escapeHtml(data.agent_1)}</div>
                        </details>
                        <details>
                            <summary>Agent 2: Builder [${escapeHtml(m.agent_2 || '')}]</summary>
                            <div class="agent-content">${escapeHtml(data.agent_2)}</div>
                        </details>
                        <details>
                            <summary>Agent 3: Critic [${escapeHtml(m.agent_3 || '')}]</summary>
                            <div class="agent-content">${escapeHtml(data.agent_3)}</div>
                        </details>
                        <details>
                            <summary>Agent 4: Optimizer [${escapeHtml(m.agent_4 || '')}]</summary>
                            <div class="agent-content">${escapeHtml(data.agent_4)}</div>
                        </details>
                    </div>
                `;
                log.appendChild(turnDiv);
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            }

            function escapeHtml(text) {
                return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            }
        </script>
    </body>
    </html>
    """
