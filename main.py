import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())

# গ্লোবাল ভেরিয়েবল যা সফল মডেলটি ধরে রাখবে
WORKING_MODEL_NAME = None

def find_working_free_model() -> str:
    global WORKING_MODEL_NAME
    if WORKING_MODEL_NAME:
        return WORKING_MODEL_NAME

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY পাওয়া যায়নি! Render Environment Variables চেক করুন।")

    # generateContent সমর্থনকারী মডেলের তালিকা সংগ্রহ
    try:
        all_models = [
            m.name for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]
    except Exception as e:
        raise RuntimeError(f"গুগল এপিআই তালিকা লোড করতে ব্যর্থ: {str(e)}")

    # বিলিং বা পেইড মডেলগুলো বাদ দেওয়া (omni, pro, ultra)
    safe_candidates = [
        m for m in all_models
        if not any(blocked in m.lower() for blocked in ['omni', 'pro', 'ultra', 'preview-02'])
    ]

    # ফ্রি ফ্ল্যাশ ও লাইট মডেলগুলোকে অগ্রাধিকার দেওয়া
    priority_order = ['2.5-flash', '2.0-flash', 'flash-lite', '1.5-flash-8b', 'flash']
    sorted_candidates = []
    
    for p in priority_order:
        for m in safe_candidates:
            if p in m.lower() and m not in sorted_candidates:
                sorted_candidates.append(m)

    # বাকি সব নিরাপদ মডেল যুক্ত করা
    for m in safe_candidates:
        if m not in sorted_candidates:
            sorted_candidates.append(m)

    # একের পর এক টেস্ট করে ফ্রি কোটা সচল থাকা প্রথম মডেলটি লক করা
    last_err = ""
    for model_name in sorted_candidates:
        try:
            test_m = genai.GenerativeModel(model_name)
            res = test_m.generate_content("hi")
            if res and res.text:
                WORKING_MODEL_NAME = model_name
                print(f"[SUCCESS] Free active model locked: {WORKING_MODEL_NAME}")
                return WORKING_MODEL_NAME
        except Exception as err:
            last_err = str(err)
            continue

    raise RuntimeError(f"কোনো ফ্রি মডেল পাওয়া যায়নি। সর্বশেষ ত্রুটি: {last_err}")

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    prompt: str
    history: List[Message] = []

@app.get("/ping")
def ping():
    return {"status": "alive", "active_model": WORKING_MODEL_NAME or "standby"}

async def run_agent(role_instruction: str, context: str, user_input: str) -> str:
    loop = asyncio.get_event_loop()
    # সচল মডেল খুঁজে নেওয়া
    active_model_name = await loop.run_in_executor(None, find_working_free_model)
    model = genai.GenerativeModel(active_model_name)

    prompt = f"SYSTEM ROLE: {role_instruction}\n\nCONVERSATION CONTEXT:\n{context}\n\nCURRENT TASK:\n{user_input}"
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    return response.text

@app.post("/api/chat")
async def chat_multi_agent(payload: ChatPayload):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="প্রম্পট খালি রাখা যাবে না")

    history_context = "\n".join([f"{m.role.upper()}: {m.content}" for m in payload.history[-6:]])

    try:
        # এজেন্ট ১: Orchestrator
        a1_plan = await run_agent(
            "You are Agent 1 (Orchestrator). Break down the user problem into clear technical sub-goals.",
            history_context, payload.prompt
        )

        # এজেন্ট ২: Architect & Builder
        a2_build = await run_agent(
            "You are Agent 2 (Builder/Coder). Generate the core structural logic, implementation, or solution based on Agent 1's plan.",
            history_context, a1_plan
        )

        # এজেন্ট ৩: Critic & Security QA
        a3_critic = await run_agent(
            "You are Agent 3 (Critic/QA). Aggressively spot bugs, missing edge cases, security issues, or logical flaws in Agent 2's output.",
            history_context, a2_build
        )

        # এজেন্ট ৪: Optimizer & Refiner
        a4_refine = await run_agent(
            "You are Agent 4 (Optimizer). Fix all issues found by Agent 3 and polish the implementation to production standard.",
            history_context, f"Code:\n{a2_build}\n\nCritique:\n{a3_critic}"
        )

        # এজেন্ট ৫: Executive Presenter
        a5_final = await run_agent(
            "You are Agent 5 (Presenter). Prepare the final response for the user. Deliver an impeccably organized, crystal-clear deliverable. No robotic introductions.",
            history_context, a4_refine
        )

        return {
            "model_used": WORKING_MODEL_NAME,
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
            .final-title { font-size: 1rem; font-weight: 700; color: #7ee787; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
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

                turnDiv.innerHTML = `
                    <div class="user-msg">${escapeHtml(prompt)}</div>
                    
                    <div class="final-card">
                        <div class="final-title">
                            <span>✓ Agent 5: Verified & Structured Output</span>
                            <span class="model-badge">${escapeHtml(data.model_used || 'Active')}</span>
                        </div>
                        <div class="output-text">${escapeHtml(data.final_output)}</div>
                    </div>

                    <div class="agent-tray">
                        <details>
                            <summary>Agent 1: Orchestrator (Strategy & Breakdown)</summary>
                            <div class="agent-content">${escapeHtml(data.agent_1)}</div>
                        </details>
                        <details>
                            <summary>Agent 2: Builder (Structural Draft)</summary>
                            <div class="agent-content">${escapeHtml(data.agent_2)}</div>
                        </details>
                        <details>
                            <summary>Agent 3: Critic (QA & Security Scan)</summary>
                            <div class="agent-content">${escapeHtml(data.agent_3)}</div>
                        </details>
                        <details>
                            <summary>Agent 4: Optimizer (Refined Deliverable)</summary>
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
