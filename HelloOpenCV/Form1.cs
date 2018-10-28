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

            net = DnnInvoke.ReadNetFromTensorflow(@"Properties\frozen_inference_graph_141 14-51-46-798.pb", @"Properties\graph.pbtxt");

            if(net == null)
            {
                throw new Exception("Error");
            }

            Application.Idle += new EventHandler(delegate
            {
                if(img != null)
                {
                    imageBox1.Image = ProcessImageUseDnn(img);
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

        private Image<Bgr, byte> ProcessImageUseDnn(Image<Bgr, byte> img)
        {
            Mat frame = img.Mat;

            int inWidth = 300;
            int inHeight = 300;
            float WHRatio = inWidth / (float)inHeight;
            String[] classNames = new String[]
            {
                "background",
                "face"
            };

            Size frame_size = frame.Size;
            Size cropSize = new Size();

            if (frame_size.Width / (float)frame_size.Height > WHRatio)
            {
                cropSize = new Size(
                    (int)(frame_size.Height * WHRatio),
                    frame_size.Height
                    );
            }
            else
            {
                cropSize = new Size(
                    frame_size.Width,
                   (int)(frame_size.Width / WHRatio)
                   );
            }

            Rectangle crop = new Rectangle(new Point((frame_size.Width -cropSize.Width) / 2,
                (frame_size.Height - cropSize.Height) / 2),
                cropSize);

            Mat inputBlob = DnnInvoke.BlobFromImage(frame, 0.00784, new Size(300, 300), new MCvScalar(127.5, 127.5, 127.5), true, false);
            net.SetInput(inputBlob, "image_tensor");
            Mat output = net.Forward("detection_out");

          //Mat detectionMat = new Mat(output.size[2], output.size[3], Emgu.CV.CvEnum.DepthType.Cv32F, output.ptr<float>());
            Mat detectionMat = new Mat(output.Size.Width, output.Size.Height, Emgu.CV.CvEnum.DepthType.Cv32F, 0);

            //frame = frame(crop);
            float confidenceThreshold = 0.20f;
            for (int i = 0; i < detectionMat.Rows; i++)
            {
                float confidence = detectionMat.GetData(i, 2 )[0];

                if (confidence > confidenceThreshold)
                {
                    int objectClass = (int)(detectionMat.GetData(i, 1)[0]);

                    int xLeftBottom = (int)(detectionMat.GetData(i, 3)[0] * frame.Cols);
                    int yLeftBottom = (int)(detectionMat.GetData(i, 4)[0] * frame.Rows);
                    int xRightTop = (int)(detectionMat.GetData(i, 5)[0] * frame.Cols);
                    int yRightTop = (int)(detectionMat.GetData(i, 6)[0] * frame.Rows);
                    
                    String conf = Convert.ToString(confidence);

                    Rectangle object1 = new Rectangle((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

                    CvInvoke.Rectangle(frame, object1, new MCvScalar(0, 255, 0), 2);
                    String label = classNames[objectClass] + ": " + conf;
                    int baseLine = 0;
                    Size labelSize = CvInvoke.GetTextSize(label, FontFace.HersheySimplex, 0.5, 1, ref baseLine);
                    CvInvoke.Rectangle(frame, new Rectangle(new Point(xLeftBottom, yLeftBottom - labelSize.Height),
                        new Size(labelSize.Width, labelSize.Height + baseLine)),
                        new MCvScalar(0, 255, 0));
                    CvInvoke.PutText(frame, label, new Point(xLeftBottom, yLeftBottom),
                        FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 0));
                }
            }

            return img;
        }
    }
}
