import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

# Render Environment Variables থেকে কী রিড করবে
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
#model = genai.GenerativeModel("gemini-1.5-flash")
model = genai.GenerativeModel("gemini-1.5-flash-latest")
class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    prompt: str
    history: List[Message] = []

# UptimeRobot হেলথ চেক রুট
@app.get("/ping")
def ping():
    return {"status": "alive"}

async def run_agent(role_instruction: str, context: str, user_input: str) -> str:
    prompt = f"SYSTEM ROLE: {role_instruction}\n\nCONVERSATION CONTEXT:\n{context}\n\nCURRENT TASK:\n{user_input}"
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    return response.text

@app.post("/api/chat")
async def chat_multi_agent(payload: ChatPayload):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="প্রম্পট খালি রাখা যাবে না")

    history_context = "\n".join([f"{m.role.upper()}: {m.content}" for m in payload.history[-6:]])

    try:
        a1_plan = await run_agent(
            "You are Agent 1 (Orchestrator). Break down the user problem into clear technical sub-goals.",
            history_context, payload.prompt
        )

        a2_build = await run_agent(
            "You are Agent 2 (Builder/Coder). Generate the core structural logic, implementation, or solution based on Agent 1's plan.",
            history_context, a1_plan
        )

        a3_critic = await run_agent(
            "You are Agent 3 (Critic/QA). Aggressively spot bugs, missing edge cases, security issues, or logical flaws in Agent 2's output.",
            history_context, a2_build
        )

        a4_refine = await run_agent(
            "You are Agent 4 (Optimizer). Fix all issues found by Agent 3 and polish the implementation to production standard.",
            history_context, f"Code:\n{a2_build}\n\nCritique:\n{a3_critic}"
        )

        a5_final = await run_agent(
            "You are Agent 5 (Presenter). Prepare the final response for the user. Deliver an impeccably organized, crystal-clear deliverable. No robotic introductions.",
            history_context, a4_refine
        )

        return {
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
        <title>Autonomous 5-Agent Hub</title>
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
            .final-title { font-size: 1rem; font-weight: 700; color: #7ee787; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
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
                status.innerText = "৫টি এজেন্ট ক্রমান্বয়ে কাজ করছে...";

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
                            ✓ Agent 5: Verified & Structured Output
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
