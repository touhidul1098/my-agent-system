import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY.strip()) if GROQ_API_KEY else None

# ৫টি এজেন্টের জন্য স্বয়ংক্রিয়ভাবে নির্বাচিত ৫টি সচল মডেলের তালিকা
WORKING_MODELS: List[str] = []

def discover_working_models(target_count: int = 5) -> List[str]:
    """Groq API থেকে সক্রিয় মডেলসমূহ সংগ্রহ করে টেস্ট করে প্রথম ৫টি সচল মডেল লক করে।"""
    global WORKING_MODELS
    if len(WORKING_MODELS) >= target_count:
        return WORKING_MODELS

    if not client:
        raise RuntimeError("GROQ_API_KEY পাওয়া যায়নি! Environment Variables চেক করুন।")

    try:
        # Groq এর লাইভ মডেল তালিকা নেওয়া
        models_data = client.models.list().data
        
        # Whisper, Audio বাদে কেবল Chat/LLM মডেলগুলো ফিল্টার করা
        available_ids = [
            m.id for m in models_data 
            if getattr(m, "active", True) and not any(blocked in m.id.lower() for blocked in ["whisper", "tts", "audio", "embed"])
        ]
    except Exception as e:
        raise RuntimeError(f"Groq API থেকে মডেল তালিকা লোড করতে ব্যর্থ: {str(e)}")

    verified_models = []
    print("[INFO] সচল মডেল টেস্ট ও শনাক্তকরণ শুরু হচ্ছে...")

    for model_id in available_ids:
        try:
            # ছোট্ট টেস্ট রিকোয়েস্ট দিয়ে কোটা ও সক্রিয়তা যাচাই
            res = client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model=model_id,
                max_tokens=5
            )
            if res and res.choices:
                verified_models.append(model_id)
                print(f"[ACTIVE] মডেল প্রস্তুত: {model_id}")
                if len(verified_models) == target_count:
                    break
        except Exception:
            # মডেল ডিকমিশন বা রেট-লিমিট থাকলে বাদ দিয়ে পরেরটি দেখবে
            continue

    if not verified_models:
        raise RuntimeError("কোনো সচল ও টেস্টে উত্তীর্ণ Groq মডেল পাওয়া যায়নি।")

    # যদি ৫টির কম সচল মডেল পাওয়া যায়, তবে প্রাপ্ত মডেলগুলো রিপিট করে ৫টি পূরণ করবে
    while len(verified_models) < target_count:
        verified_models.append(verified_models[len(verified_models) % len(verified_models)])

    WORKING_MODELS = verified_models[:target_count]
    print(f"[SUCCESS] ৫টি এজেন্টের জন্য নির্বাচিত মডেলসমূহ: {WORKING_MODELS}")
    return WORKING_MODELS

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
        "assigned_models": WORKING_MODELS if len(WORKING_MODELS) == 5 else "Discovering on next run"
    }

async def run_agent(role_instruction: str, context: str, user_input: str, model_name: str) -> str:
    loop = asyncio.get_event_loop()
    
    messages = [
        {"role": "system", "content": f"{role_instruction}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": user_input}
    ]

    def call_groq():
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    return await loop.run_in_executor(None, call_groq)

@app.post("/api/chat")
async def chat_multi_agent(payload: ChatPayload):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="প্রম্পট খালি রাখা যাবে না")

    loop = asyncio.get_event_loop()
    # ৫টি সচল মডেল নিশ্চিত করা
    models = await loop.run_in_executor(None, discover_working_models)

    history_context = "\n".join([f"{m.role.upper()}: {m.content}" for m in payload.history[-6:]])

    try:
        # এজেন্ট ১: Orchestrator (Model 1)
        a1_plan = await run_agent(
            "You are Agent 1 (Orchestrator). Break down the user problem into clear technical sub-goals.",
            history_context, payload.prompt, models[0]
        )

        # এজেন্ট ২: Architect & Builder (Model 2)
        a2_build = await run_agent(
            "You are Agent 2 (Builder/Coder). Generate the core structural logic, implementation, or solution based on Agent 1's plan.",
            history_context, a1_plan, models[1]
        )

        # এজেন্ট ৩: Critic & Security QA (Model 3)
        a3_critic = await run_agent(
            "You are Agent 3 (Critic/QA). Aggressively spot bugs, missing edge cases, security issues, or logical flaws in Agent 2's output.",
            history_context, a2_build, models[2]
        )

        # এজেন্ট ৪: Optimizer & Refiner (Model 4)
        a4_refine = await run_agent(
            "You are Agent 4 (Optimizer). Fix all issues found by Agent 3 and polish the implementation to production standard.",
            history_context, f"Code:\n{a2_build}\n\nCritique:\n{a3_critic}", models[3]
        )

        # এজেন্ট ৫: Executive Presenter (Model 5)
        a5_final = await run_agent(
            "You are Agent 5 (Presenter). Prepare the final response for the user. Deliver an impeccably organized, crystal-clear deliverable. No robotic introductions.",
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
                    const data = await res.json();
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
