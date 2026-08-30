import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Ensure project root in python path
sys.path.insert(0, os.path.abspath("."))

# Use PYTHON env var if set, otherwise the interpreter running this app
PYTHON_EXE = os.environ.get("PYTHON", sys.executable)

class DrivingLabApp:
    """GUI Launcher App for Autonomous Driving Simulator & Agent Benchmark Lab."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AUTONOMOUS DRIVING LAB - MetaDrive Environment")
        self.root.geometry("640x580")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")

        self._create_styles()
        self._build_ui()

    def _create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1E1E2E", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#89B4FA", background="#1E1E2E")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "italic"), foreground="#A6ADC8", background="#1E1E2E")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8)

    def _build_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#1E1E2E")
        header_frame.pack(fill="x", pady=15, padx=20)

        title_lbl = ttk.Label(header_frame, text="🏎️ AUTONOMOUS DRIVING LAB", style="Header.TLabel")
        title_lbl.pack()

        subtitle_lbl = ttk.Label(
            header_frame,
            text="Comparative Evaluation: Human vs. Small-LLM vs. RL Agents in MetaDrive",
            style="SubHeader.TLabel",
        )
        subtitle_lbl.pack(pady=4)

        # Main Control Card
        card = tk.Frame(self.root, bg="#313244", bd=2, relief="groove")
        card.pack(fill="both", expand=True, padx=25, pady=10)

        # Seed Selection
        seed_frame = tk.Frame(card, bg="#313244")
        seed_frame.pack(fill="x", padx=20, pady=12)

        ttk.Label(seed_frame, text="Scenario Seed / Identifier:", font=("Segoe UI", 11, "bold"), background="#313244").pack(side="left")

        self.seed_var = tk.StringVar(value="2037")
        seed_combo = ttk.Combobox(
            seed_frame,
            textvariable=self.seed_var,
            values=["2037", "2000", "2001", "2002", "2003", "1000", "1100"],
            width=10,
            font=("Segoe UI", 11),
        )
        seed_combo.pack(side="right")

        # Action Buttons Frame
        btn_frame = tk.Frame(card, bg="#313244")
        btn_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Play Human Button
        btn_human = tk.Button(
            btn_frame,
            text="👤 PLAY HUMAN DRIVER (Live 3D WASD)",
            bg="#A6E3A1",
            fg="#11111B",
            font=("Segoe UI", 11, "bold"),
            activebackground="#94E2D5",
            cursor="hand2",
            command=lambda: self._launch_agent("human"),
        )
        btn_human.pack(fill="x", pady=6)

        # Watch RL Button
        btn_rl = tk.Button(
            btn_frame,
            text="🤖 WATCH RL AGENT (SAC Policy Live 3D)",
            bg="#89B4FA",
            fg="#11111B",
            font=("Segoe UI", 11, "bold"),
            activebackground="#74C7EC",
            cursor="hand2",
            command=lambda: self._launch_agent("rl"),
        )
        btn_rl.pack(fill="x", pady=6)

        # Watch LLM Button
        btn_llm = tk.Button(
            btn_frame,
            text="🧠 WATCH LLM AGENT (High-Level Planner Live 3D)",
            bg="#F9E2AF",
            fg="#11111B",
            font=("Segoe UI", 11, "bold"),
            activebackground="#FAB387",
            cursor="hand2",
            command=lambda: self._launch_agent("llm"),
        )
        btn_llm.pack(fill="x", pady=6)

        # Preview Scenario Button
        btn_preview = tk.Button(
            btn_frame,
            text="👁️ PREVIEW SCENARIO ROAD GEOMETRY",
            bg="#CBA6F7",
            fg="#11111B",
            font=("Segoe UI", 10, "bold"),
            activebackground="#B4BEFE",
            cursor="hand2",
            command=self._launch_preview,
        )
        btn_preview.pack(fill="x", pady=6)

        # Separator Line
        ttk.Separator(card, orient="horizontal").pack(fill="x", padx=20, pady=10)

        # Research Experiment Buttons
        exp_frame = tk.Frame(card, bg="#313244")
        exp_frame.pack(fill="x", padx=20, pady=5)

        btn_batch = tk.Button(
            exp_frame,
            text="📊 RUN 20-SEED BATCH BENCHMARK",
            bg="#F38BA8",
            fg="#11111B",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=self._launch_batch,
        )
        btn_batch.pack(side="left", fill="x", expand=True, padx=4)

        btn_analyze = tk.Button(
            exp_frame,
            text="📈 ANALYZE STATISTICS & PLOTS",
            bg="#94E2D5",
            fg="#11111B",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=self._launch_analyze,
        )
        btn_analyze.pack(side="right", fill="x", expand=True, padx=4)

        # Footer Status Bar
        footer = tk.Frame(self.root, bg="#181825", height=30)
        footer.pack(fill="x", side="bottom")
        self.status_lbl = tk.Label(footer, text="Ready. Select a mode and click PLAY to launch live 3D MetaDrive.", bg="#181825", fg="#A6ADC8", font=("Segoe UI", 9))
        self.status_lbl.pack(pady=4)

    def _get_seed(self) -> int:
        try:
            return int(self.seed_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Seed", "Please enter a valid numeric seed (e.g. 2037)")
            return 2037

    def _launch_agent(self, agent_type: str):
        seed = self._get_seed()
        self.status_lbl.config(text=f"Launching MetaDrive 3D Window for {agent_type.upper()} on Seed {seed}...")
        cmd = [PYTHON_EXE, "run.py", "--agent", agent_type, "--seed", str(seed)]
        subprocess.Popen(cmd, cwd=os.path.abspath("."))

    def _launch_preview(self):
        seed = self._get_seed()
        self.status_lbl.config(text=f"Launching Scenario Preview for Seed {seed}...")
        cmd = [PYTHON_EXE, "run.py", "--agent", "human", "--seed", str(seed)]
        subprocess.Popen(cmd)

    def _launch_batch(self):
        self.status_lbl.config(text="Launching Batch Evaluation Benchmark across test seeds...")
        cmd = [PYTHON_EXE, "evaluate.py", "--seeds", "test", "--count", "5"]
        subprocess.Popen(cmd)

    def _launch_analyze(self):
        self.status_lbl.config(text="Running statistical hypothesis testing and generating plots...")
        cmd = [PYTHON_EXE, "analyze.py"]
        proc = subprocess.Popen(cmd)
        proc.wait()
        messagebox.showinfo("Analysis Complete", "Statistical report CSVs and visual plots have been generated in results/")
        self.status_lbl.config(text="Analysis complete. Check results/ and results/plots/")

def main():
    root = tk.Tk()
    app = DrivingLabApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
