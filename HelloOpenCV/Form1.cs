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
using Emgu.CV.Dnn;

namespace HelloOpenCV
{
    public partial class Form1 : Form
    {
        private static VideoCapture cameraCapture;
        CascadeClassifier cascadeClassifier;
        Net net;

        Image<Bgr, byte> img;

        public Form1()
        {
            InitializeComponent();

            cascadeClassifier = new CascadeClassifier(@"Properties\haarcascade_frontalface_alt.xml");

            net = DnnInvoke.ReadNetFromTensorflow(@"Properties\frozen_inference_graph.pb", @"Properties\graph.pbtxt");

            if (net == null)
            {
                throw new Exception("Error");
            }

            Application.Idle += new EventHandler(delegate
            {
                if (img != null)
                {
                    //img = ProcessImage(img);
                    img = ProcessImageUseDnn(img);
                    imageBox1.Image = img;

                }
            });

            button2_Click(null, null);
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
                CvInvoke.Flip(img.Mat, img.Mat, FlipType.Horizontal);
            });
        }

        private void OpenCamera()
        {
            try
            {
                cameraCapture = new VideoCapture();
                cameraCapture.SetCaptureProperty(CapProp.FrameWidth, 300);
                cameraCapture.SetCaptureProperty(CapProp.FrameHeight, 300);
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
                float X = (face.Left + face.Right) / 2;
                float Y = (face.Top + face.Bottom) / 2;
                float W = (face.Right - face.Left) / 2;
                //float H = (face.Bottom - face.Top) / 2;
                //img.Draw(face, new Bgr(0, double.MaxValue, 0), 3);
                img.Draw(new CircleF(new PointF(X, Y), W / 3), new Bgr(0, double.MaxValue, 0), (int)W);
            }
            return img;
        }

        private Image<Bgr, byte> ProcessImageUseDnn(Image<Bgr, byte> img)
        {
            Mat frame = img.Mat;

            String[] classNames = new String[]
            {
                "",
                "hand"
            };

            Mat inputBlob = DnnInvoke.BlobFromImage(frame, 1, new Size(300, 300));

            net.SetInput(inputBlob, "image_tensor");
            Mat prob = net.Forward("detection_out");

            byte[] data = new byte[5600];
            prob.CopyTo(data);

            //draw result
            for (int i = 0; i < prob.SizeOfDimemsion[2]; i++)
            {
                var d = BitConverter.ToSingle(data, i * 28 + 8);
                if (d > 0.3)
                {
                    var idx = (int)BitConverter.ToSingle(data, i * 28 + 4);
                    var w1 = (int)(BitConverter.ToSingle(data, i * 28 + 12) * img.Width);
                    var h1 = (int)(BitConverter.ToSingle(data, i * 28 + 16) * img.Height);
                    var w2 = (int)(BitConverter.ToSingle(data, i * 28 + 20) * img.Width);
                    var h2 = (int)(BitConverter.ToSingle(data, i * 28 + 24) * img.Height);

                    double X = (w1 + w2) / 2;
                    double Y = (h1 + h2) / 2;

                    if (false)
                    {
                        continue;
                    }

                    var label = $"{classNames[idx]} {d * 100:0.00}%";

                    CvInvoke.Rectangle(img, new Rectangle(w1, h1, w2 - w1, h2 - h1), new MCvScalar(0, 255, 0), 2);

                    int baseline = 0;
                    var textSize = CvInvoke.GetTextSize(label, FontFace.HersheyTriplex, 0.5, 1, ref baseline);
                    var y = h1 - textSize.Height < 0 ? h1 + textSize.Height : h1;
                    CvInvoke.Rectangle(img, new Rectangle(w1, y - textSize.Height, textSize.Width, textSize.Height), new MCvScalar(0, 255, 0), -1);
                    CvInvoke.PutText(img, label, new Point(w1, y), FontFace.HersheyTriplex, 0.5, new Bgr(0, 0, 0).MCvScalar);
                }
            }

            return img;
        }
    }
}
