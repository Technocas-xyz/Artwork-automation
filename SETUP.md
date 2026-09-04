# Designer Agent — Setup

The website runs on a server. The artwork generation runs on **your PC**, because
it needs a real Chrome with your logged-in ChatGPT session. This small "agent"
connects your PC to the website. You only set it up once.

There are **no ports to open and no firewall changes** — your PC reaches out to
the server, never the other way around.

---

## Easiest: the packaged app (no Python)

1. Log in to the website.
2. Click **Download agent** in the header, then **Download ArtworkAgent.zip**.
3. **Unzip** it anywhere (e.g. your Desktop), open the unzipped `ArtworkAgent`
   folder, and run **ArtworkAgent.exe**. A small window opens.
   (Keep the folder together — the app needs the files beside the exe.)
4. On the website's download panel, click **Copy** next to *Your token* and paste
   it into the agent's **Agent token** box. The server URL is pre-filled.
5. Click **Sign in to ChatGPT**, log in in the browser that opens, then close it.
6. Click **Start**. The agent minimises to the system tray:
   - **green** = connected, **amber** = not signed in, **red** = error.

The website header shows **"Agent connected — Your Name"** in green once it's up.

---

## Advanced: run from Python source

Use this only if you are not using the packaged .exe. Build tools
(`pyinstaller`, `pystray`) are not needed just to run from source.

---

## One-time setup

1. **Install Python 3.11+**
   Download from <https://www.python.org/downloads/>. During install, tick
   **"Add Python to PATH"**.

2. **Get the app onto your PC**
   Unzip the folder you were given (or `git clone` it) somewhere easy, e.g.
   `C:\artwork-studio`.

3. **Create the environment and install requirements**
   Open a terminal in that folder and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   playwright install chromium
   ```

4. **Sign in to ChatGPT once**
   ```
   python login.py
   ```
   A Chrome window opens. Log in to ChatGPT normally. The session is saved to a
   local browser profile, so you stay logged in between runs. Close it when done.

5. **Paste your agent token**
   You were given a token (ask the admin; they generate it with
   `python make_agent_token.py "Your Name"`). Create a file named `.env` in the
   folder, beside `agent.py`, containing:
   ```
   SERVER_URL=https://your-server-address
   AGENT_TOKEN=agt_your_token_here
   AGENT_NAME=Your Name
   ```

---

## Every day

- **Double-click `run_agent.bat`.**
  Leave the window open while you work. When it says
  `ChatGPT session OK — logged in` and `registered`, the website header will show
  **"Agent connected — Your Name"** in green.

- Generate artwork from the website as usual. Jobs you start there run on your PC
  through this agent.

- To stop, close the window (or press `Ctrl+C`).

---

## If something goes wrong

- **Header shows "No agent running" (amber):** the agent isn't running — double-click
  `run_agent.bat`.
- **"NOT LOGGED IN" / "session expired":** run `python login.py`, sign in again,
  then start `run_agent.bat`.
- **Nothing happens after you click Generate:** make sure the agent window is open
  and shows "registered". Only one designer's agent needs to run at a time, but
  several can — jobs are shared out automatically.
