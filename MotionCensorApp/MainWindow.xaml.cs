using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Win32;

namespace MotionCensorApp
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private string selectedVideoPath;

        private void SelectVideoButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Video Files|*.mp4;*.avi;*.mkv";

            if (openFileDialog.ShowDialog() == true)
            {
                selectedVideoPath = openFileDialog.FileName;
                VideoPathTextBox.Text = selectedVideoPath;
            }
        }

        private async void CensorButton_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(selectedVideoPath))
            {
                string outputFolder = @"C:\Users\mathe\OneDrive\Desktop\Modified";
                string pythonScript = @"C:\Users\mathe\source\repos\MotionCensorApp\MotionCensorApp\transcribe_censor.py";

                if (!File.Exists(pythonScript))
                {
                    MessageBox.Show($"Python script not found at: {pythonScript}");
                    return;
                }

                // Show the processing indicator
                ProcessingProgressBar.Visibility = Visibility.Visible;
                ProcessingTextBlock.Visibility = Visibility.Visible;
                ProcessingTextBlock.Text = "🚀 Processing Started...";
                CensorButton.IsEnabled = false;
                SelectVideoButton.IsEnabled = false;
                TranscriptionTextBox.Clear();

                try
                {
                    AppendTextToTextBox("🔹 Censorship process initiated...");

                    var process = new Process
                    {
                        StartInfo = new ProcessStartInfo
                        {
                            FileName = "python",
                            Arguments = $"\"{pythonScript}\" \"{selectedVideoPath}\" \"{outputFolder}\"",
                            UseShellExecute = false,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            CreateNoWindow = true
                        }
                    };

                    process.OutputDataReceived += (processSender, args) =>
                    {
                        if (args.Data != null)
                        {
                            AppendTextToTextBox(args.Data);
                        }
                    };

                    process.ErrorDataReceived += (processSender, args) =>
                    {
                        if (args.Data != null)
                        {
                            AppendTextToTextBox($"❗ Error: {args.Data}");
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    await Task.Run(() => process.WaitForExit());

                    AppendTextToTextBox("✅ Censorship process completed!");

                    string transcriptionPath = Path.Combine(outputFolder, "transcription.json");
                    string outputPath = Path.Combine(outputFolder, "censored_output.mp4");
                    MessageBox.Show("Censorship Completed. Output saved at: " + outputPath);

                    // Predict genre after transcription
                    ProcessingTextBlock.Text = "🔍 Predicting Genre...";
                    await PredictGenre(transcriptionPath);
                }
                catch (Exception ex)
                {
                    AppendTextToTextBox($"❗ Error: {ex.Message}");
                }
                finally
                {
                    ProcessingProgressBar.Visibility = Visibility.Collapsed;
                    ProcessingTextBlock.Visibility = Visibility.Collapsed;
                    CensorButton.IsEnabled = true;
                    SelectVideoButton.IsEnabled = true;
                }
            }
            else
            {
                MessageBox.Show("Please select a video file.");
            }
        }

        private async Task PredictGenre(string transcriptionJsonPath)
        {
            string pythonScript = @"C:\Users\mathe\source\repos\MotionCensorApp\MotionCensorApp\predict_genre.py";

            if (!File.Exists(pythonScript))
            {
                MessageBox.Show($"Genre prediction script not found at: {pythonScript}");
                return;
            }

            try
            {
                AppendTextToTextBox("🔹 Genre prediction initiated...");

                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"\"{pythonScript}\" \"{transcriptionJsonPath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                process.OutputDataReceived += (sender, args) =>
                {
                    if (args.Data != null)
                    {
                        Dispatcher.Invoke(() =>
                        {
                            GenrePredictionTextBlock.Text = $"🎬 Predicted Genre: {args.Data}";
                            AppendTextToTextBox($"🎬 Predicted Genre: {args.Data}");
                        });
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                await Task.Run(() => process.WaitForExit());

                AppendTextToTextBox("✅ Genre prediction completed!");
            }
            catch (Exception ex)
            {
                AppendTextToTextBox($"❗ Error: {ex.Message}");
            }
        }


        private void AppendTextToTextBox(string text)
        {
            Dispatcher.Invoke(() =>
            {
                TranscriptionTextBox.AppendText(text + Environment.NewLine);
                TranscriptionTextBox.ScrollToEnd();
            });
        }
    }
}
