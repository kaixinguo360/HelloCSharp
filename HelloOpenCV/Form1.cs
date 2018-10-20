using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.UI;

namespace HelloOpenCV
{
    public partial class Form1 : Form
    {
        private static VideoCapture cameraCapture;
        CascadeClassifier cascadeClassifier;

        Image<Bgr, byte> img;

        public Form1()
        {
            InitializeComponent();

            cascadeClassifier = new CascadeClassifier(@"F:\Tools\OpenCV\build\etc\haarcascades\haarcascade_frontalface_alt.xml");

            Application.Idle += new EventHandler(delegate
            {
                if(img != null)
                {
                    imageBox1.Image = ProcessImage(img);
                }
            });
        }

        private void button1_Click(object sender, EventArgs e)
        {
            label1.Text = "Hello,C#!";
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "BMP文件|*.bmp|JPG文件|*.jpg|JPEG文件|*.jpeg|所有文件|*.*";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                img = new Image<Bgr, byte>(openFileDialog.FileName);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenCamera();
            Application.Idle += new EventHandler(delegate
            {
                img = cameraCapture.QueryFrame().ToImage<Bgr, byte>();
            });
        }

        private void OpenCamera()
        {
            try
            {
                cameraCapture = new VideoCapture();
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
                return;
            }
        }

        private Image<Bgr, byte> ProcessImage(Image<Bgr, byte> img)
        {
            var grayframe = img.Convert<Gray, byte>();
            var faces = cascadeClassifier.DetectMultiScale(grayframe);

            foreach (var face in faces)
            {
                img.Draw(face, new Bgr(0, double.MaxValue, 0), 3);
            }
            return img;
        }
    }
}
