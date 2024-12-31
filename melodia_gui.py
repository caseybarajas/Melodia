# melodia_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import logging
import queue
import subprocess
import sys
import os

class MelodiaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Melodia Music Generation")
        self.root.geometry("1024x768")  # Larger default size
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use modern theme
        
        # Configure styles
        self.setup_styles()
        
        # Create queue for log messages
        self.log_queue = queue.Queue()
        self.setup_logging()
        
        # Create main container with padding
        self.main_container = ttk.Frame(self.root, padding="20", style='Main.TFrame')
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create header
        self.setup_header()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container, style='Custom.TNotebook')
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create tabs with padding
        self.training_tab = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.generation_tab = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.generation_tab, text="Generation")
        
        # Set up tabs
        self.setup_training_tab()
        self.setup_generation_tab()
        
        # Set up log display
        self.setup_log_display()
        
        # Configure grid weights for scaling
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(1, weight=3)  # Notebook takes more space
        self.main_container.rowconfigure(2, weight=1)  # Log takes less space
        
        # Configure tab grid weights
        self.training_tab.columnconfigure(1, weight=1)  # Entry fields expand
        self.generation_tab.columnconfigure(1, weight=1)
    
    def setup_styles(self):
        """Configure custom styles for widgets"""
        # Custom colors
        bg_color = "#f0f0f0"
        accent_color = "#2c3e50"
        highlight_color = "#3498db"
        
        # Frame styles
        self.style.configure('Main.TFrame', background=bg_color)
        self.style.configure('Tab.TFrame', background=bg_color)
        
        # Label styles
        self.style.configure('Header.TLabel',
                           font=('Helvetica', 24, 'bold'),
                           foreground=accent_color,
                           background=bg_color)
        
        self.style.configure('Subheader.TLabel',
                           font=('Helvetica', 12),
                           foreground=accent_color,
                           background=bg_color)
        
        # Button styles
        self.style.configure('Action.TButton',
                           font=('Helvetica', 11),
                           padding=10)
        
        self.style.configure('Browse.TButton',
                           padding=5)
        
        # Entry styles
        self.style.configure('Custom.TEntry',
                           padding=5)
        
        # Notebook styles
        self.style.configure('Custom.TNotebook',
                           background=bg_color,
                           padding=5)
        
        self.style.configure('Custom.TNotebook.Tab',
                           padding=(10, 5),
                           font=('Helvetica', 10))
        
        # Labelframe styles
        self.style.configure('Custom.TLabelframe',
                           background=bg_color,
                           padding=15)
        
        self.style.configure('Custom.TLabelframe.Label',
                           font=('Helvetica', 11, 'bold'),
                           foreground=accent_color,
                           background=bg_color)
    
    def setup_header(self):
        """Set up the header section"""
        header_frame = ttk.Frame(self.main_container, style='Main.TFrame')
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Title
        title_label = ttk.Label(header_frame,
                              text="Melodia Music Generation",
                              style='Header.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame,
                                 text="Create and train AI music models",
                                 style='Subheader.TLabel')
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
    
    def setup_training_tab(self):
        """Set up the training tab with improved layout"""
        # Directory selection frame
        dir_frame = ttk.LabelFrame(self.training_tab,
                                 text="Directories",
                                 style='Custom.TLabelframe')
        dir_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E),
                      padx=5, pady=(0, 15))
        dir_frame.columnconfigure(1, weight=1)
        
        # Data directory
        ttk.Label(dir_frame, text="Training Data:").grid(row=0, column=0,
                                                        sticky=tk.W, padx=5, pady=5)
        self.data_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.data_dir_var,
                 style='Custom.TEntry').grid(row=0, column=1,
                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Button(dir_frame, text="Browse",
                  command=self.browse_data_dir,
                  style='Browse.TButton').grid(row=0, column=2, padx=5)
        
        # Model directory
        ttk.Label(dir_frame, text="Save Models:").grid(row=1, column=0,
                                                      sticky=tk.W, padx=5, pady=5)
        self.model_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.model_dir_var,
                 style='Custom.TEntry').grid(row=1, column=1,
                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Button(dir_frame, text="Browse",
                  command=self.browse_model_dir,
                  style='Browse.TButton').grid(row=1, column=2, padx=5)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(self.training_tab,
                                   text="Training Parameters",
                                   style='Custom.TLabelframe')
        param_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E),
                        padx=5, pady=15)
        param_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(3, weight=1)
        
        # Parameter grid with proper spacing
        params = [
            ("Batch Size:", self.batch_size_var, "32", 0),
            ("Epochs:", self.epochs_var, "100", 1),
            ("Learning Rate:", self.lr_var, "0.001", 2),
            ("Validation Split:", self.val_split_var, "0.1", 3)
        ]
        
        for i, (label, var, default, row) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=row//2, column=row%2*2,
                                                  sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default)
            setattr(self, var._name, var)
            ttk.Entry(param_frame, textvariable=var,
                     style='Custom.TEntry', width=15).grid(row=row//2,
                                                         column=row%2*2+1,
                                                         sticky=(tk.W, tk.E),
                                                         padx=5, pady=5)
        
        # Controls frame
        control_frame = ttk.Frame(self.training_tab)
        control_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        self.train_button = ttk.Button(control_frame,
                                     text="Start Training",
                                     command=self.start_training,
                                     style='Action.TButton')
        self.train_button.grid(row=0, column=0, padx=10)
        
        self.stop_button = ttk.Button(control_frame,
                                    text="Stop Training",
                                    command=self.stop_training,
                                    state=tk.DISABLED,
                                    style='Action.TButton')
        self.stop_button.grid(row=0, column=1, padx=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.training_tab,
                                          mode='determinate',
                                          variable=self.progress_var)
        self.progress_bar.grid(row=3, column=0, columnspan=3,
                             sticky=(tk.W, tk.E), padx=5, pady=(10, 0))

    def setup_generation_tab(self):
        """Set up the generation tab with improved layout"""
        # Model selection frame
        model_frame = ttk.LabelFrame(self.generation_tab,
                                   text="Model Selection",
                                   style='Custom.TLabelframe')
        model_frame.grid(row=0, column=0, columnspan=3,
                        sticky=(tk.W, tk.E), padx=5, pady=(0, 15))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0,
                                                      sticky=tk.W, padx=5, pady=5)
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_path_var,
                 style='Custom.TEntry').grid(row=0, column=1,
                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Button(model_frame, text="Browse",
                  command=self.browse_model_path,
                  style='Browse.TButton').grid(row=0, column=2, padx=5)
        
        # Generation parameters frame
        gen_frame = ttk.LabelFrame(self.generation_tab,
                                 text="Generation Parameters",
                                 style='Custom.TLabelframe')
        gen_frame.grid(row=1, column=0, columnspan=3,
                      sticky=(tk.W, tk.E), padx=5, pady=15)
        gen_frame.columnconfigure(1, weight=1)
        gen_frame.columnconfigure(3, weight=1)
        
        # Parameters grid
        params = [
            ("Number of Samples:", self.num_samples_var, "1", 0),
            ("Temperature:", self.temperature_var, "1.0", 1),
            ("Style:", self.style_var, "", 2),
            ("Key:", self.key_var, "", 3)
        ]
        
        for i, (label, var, default, row) in enumerate(params):
            ttk.Label(gen_frame, text=label).grid(row=row//2, column=row%2*2,
                                               sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default)
            setattr(self, var._name, var)
            
            if row in [2, 3]:  # Style and Key use comboboxes
                values = ('classical', 'jazz', 'folk', 'blues') if row == 2 else \
                        ('C', 'G', 'D', 'A', 'E', 'B', 'F#', 'F', 'Bb', 'Eb', 'Ab')
                combo = ttk.Combobox(gen_frame, textvariable=var,
                                   values=values, width=15)
                combo.grid(row=row//2, column=row%2*2+1,
                          sticky=(tk.W, tk.E), padx=5, pady=5)
            else:  # Others use entry fields
                ttk.Entry(gen_frame, textvariable=var,
                         style='Custom.TEntry', width=15).grid(row=row//2,
                                                             column=row%2*2+1,
                                                             sticky=(tk.W, tk.E),
                                                             padx=5, pady=5)
        
        # Output directory frame
        output_frame = ttk.LabelFrame(self.generation_tab,
                                    text="Output",
                                    style='Custom.TLabelframe')
        output_frame.grid(row=2, column=0, columnspan=3,
                         sticky=(tk.W, tk.E), padx=5, pady=(0, 15))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Save To:").grid(row=0, column=0,
                                                    sticky=tk.W, padx=5, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_dir_var,
                 style='Custom.TEntry').grid(row=0, column=1,
                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Button(output_frame, text="Browse",
                  command=self.browse_output_dir,
                  style='Browse.TButton').grid(row=0, column=2, padx=5)
        
        # Generation button
        self.generate_button = ttk.Button(self.generation_tab,
                                        text="Generate Music",
                                        command=self.generate_music,
                                        style='Action.TButton')
        self.generate_button.grid(row=3, column=0, columnspan=3,
                                pady=20)

    def setup_log_display(self):
        """Set up the log display area with improved styling"""
        log_frame = ttk.LabelFrame(self.main_container,
                                 text="Activity Log",
                                 style='Custom.TLabelframe')
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S),
                      padx=5, pady=(20, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Create text widget with custom font and colors
        self.log_text = tk.Text(log_frame,
                               height=8,
                               wrap=tk.WORD,
                               font=('Consolas', 10),
                               bg='#ffffff',
                               fg='#2c3e50')
        
        # Create scrollbar with custom style
        scrollbar = ttk.Scrollbar(log_frame,
                                orient=tk.VERTICAL,
                                command=self.log_text.yview)
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Grid layout for text and scrollbar
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S),
                          padx=(5, 0), pady=5)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(0, 5), pady=5)
        
        # Start log monitoring
        self.root.after(100, self.check_log_queue)
    
    def check_log_queue(self):
        """Check for new log messages"""
        while True:
            try:
                record = self.log_queue.get_nowait()
                msg = self.format_log_message(record)
                
                # Add timestamp in gray
                self.log_text.insert(tk.END, msg + '\n', record.levelname.lower())
                
                # Auto-scroll to bottom
                self.log_text.see(tk.END)
                
                # Configure tag colors for different log levels
                self.log_text.tag_config('INFO', foreground='#2c3e50')
                self.log_text.tag_config('WARNING', foreground='#f39c12')
                self.log_text.tag_config('ERROR', foreground='#c0392b')
                self.log_text.tag_config('DEBUG', foreground='#27ae60')
                
            except queue.Empty:
                break
        self.root.after(100, self.check_log_queue)
    
    def format_log_message(self, record):
        """Format log message with timestamp and level"""
        return f"[{record.asctime.split('.')[0]}] {record.levelname}: {record.getMessage()}"
    
    def browse_data_dir(self):
        """Browse for training data directory"""
        directory = filedialog.askdirectory(title="Select Training Data Directory")
        if directory:
            self.data_dir_var.set(directory)
            logging.info(f"Selected training data directory: {directory}")
    
    def browse_model_dir(self):
        """Browse for model save directory"""
        directory = filedialog.askdirectory(title="Select Model Save Directory")
        if directory:
            self.model_dir_var.set(directory)
            logging.info(f"Selected model save directory: {directory}")
    
    def browse_model_path(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.h5;*.keras")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            logging.info(f"Selected model file: {file_path}")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
            logging.info(f"Selected output directory: {directory}")
    
    def start_training(self):
        """Start the training process"""
        if not self.validate_training_inputs():
            return
        
        # Update UI state
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        
        # Log training start
        logging.info("Starting training process...")
        logging.info(f"Batch Size: {self.batch_size_var.get()}")
        logging.info(f"Epochs: {self.epochs_var.get()}")
        logging.info(f"Learning Rate: {self.lr_var.get()}")
        
        # Create training command
        cmd = [
            sys.executable,
            "train.py",
            "--data_dir", self.data_dir_var.get(),
            "--model_dir", self.model_dir_var.get(),
            "--batch_size", self.batch_size_var.get(),
            "--epochs", self.epochs_var.get(),
            "--learning_rate", self.lr_var.get(),
            "--validation_split", self.val_split_var.get()
        ]
        
        # Start training in separate thread
        self.training_thread = threading.Thread(target=self.run_training, args=(cmd,))
        self.training_thread.daemon = True  # Thread will be killed if app closes
        self.training_thread.start()
    
    def run_training(self, cmd):
        """Run training process with progress updates"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            total_epochs = int(self.epochs_var.get())
            current_epoch = 0
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Update progress bar if epoch information is found
                    if "Epoch" in output:
                        try:
                            current_epoch = int(output.split("/")[0].split()[-1])
                            progress = (current_epoch / total_epochs) * 100
                            self.progress_var.set(progress)
                        except (ValueError, IndexError):
                            pass
                    logging.info(output.strip())
            
            rc = process.poll()
            if rc != 0:
                error_output = process.stderr.read()
                logging.error(f"Training failed with return code {rc}")
                logging.error(error_output)
                messagebox.showerror("Error", "Training failed. Check the log for details.")
            else:
                logging.info("Training completed successfully")
                messagebox.showinfo("Success", "Training completed successfully!")
                self.progress_var.set(100)
        
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            messagebox.showerror("Error", f"Training error: {str(e)}")
        
        finally:
            # Reset UI state
            self.root.after(0, self.reset_training_buttons)
    
    def generate_music(self):
        """Generate music samples"""
        if not self.validate_generation_inputs():
            return
        
        # Update UI state
        self.generate_button.config(state=tk.DISABLED)
        
        # Log generation start
        logging.info("Starting music generation...")
        logging.info(f"Number of samples: {self.num_samples_var.get()}")
        logging.info(f"Temperature: {self.temperature_var.get()}")
        if self.style_var.get():
            logging.info(f"Style: {self.style_var.get()}")
        if self.key_var.get():
            logging.info(f"Key: {self.key_var.get()}")
        
        # Create generation command
        cmd = [
            sys.executable,
            "generate.py",
            "--model_path", self.model_path_var.get(),
            "--output_dir", self.output_dir_var.get(),
            "--num_samples", self.num_samples_var.get(),
            "--temperature", self.temperature_var.get()
        ]
        
        # Add optional parameters
        if self.style_var.get():
            cmd.extend(["--style", self.style_var.get()])
        if self.key_var.get():
            cmd.extend(["--key", self.key_var.get()])
        
        # Start generation in separate thread
        self.generation_thread = threading.Thread(target=self.run_generation, args=(cmd,))
        self.generation_thread.daemon = True
        self.generation_thread.start()

def main():
    root = tk.Tk()
    app = MelodiaGUI(root)
    
    # Set window icon if available
    try:
        icon_path = Path(__file__).parent / "assets" / "melodia_icon.ico"
        if icon_path.exists():
            root.iconbitmap(str(icon_path))
    except:
        pass
    
    # Center window on screen
    window_width = 1024
    window_height = 768
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()