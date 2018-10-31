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

        private Net net;

        private Image<Bgr, byte> img;

        private bool showBoxs = false;
        private bool showPoints = false;
        private bool showKeyPoint = false;


        ////////////////
        /// 窗口操作 ///
        ////////////////

        public Form1()
        {
            InitializeComponent();

            net = DnnInvoke.ReadNetFromTensorflow(@"Properties\frozen_inference_graph.pb", @"Properties\graph.pbtxt");  //载入神经网络

            if (net == null) //如果载入失败
            {
                throw new Exception("Error"); //报错
            }

            Application.Idle += new EventHandler(delegate  //于空闲状态时执行以下代码
            {
                if (img != null)                                //如果摄像头有传入图像
                {                                               //则依次经过以下图像处理方法
                    img = ProcessImageUseDnn(img);              //  1.DNN网络手部检测方法
                    if (showPoints) img = DrawPoints(img);      //  2.检测点绘制方法
                    if (showKeyPoint) img = DrawKeyPoint(img);  //  3.关键点绘制方法
                    imageBox1.Image = img;                      //最后输出图像到 imageBox1

                }
            });
        }
        
        /***** 运行按钮 点击事件 ****/
        private void button1_Click(object sender, EventArgs e)
        {
            OpenCamera();
            Application.Idle += new EventHandler(delegate    //于空闲状态时执行以下代码
            {
                img = cameraCapture.QueryFrame().ToImage<Bgr, byte>(); //从摄像头获取一帧图像
                CvInvoke.Flip(img.Mat, img.Mat, FlipType.Horizontal);  //水平反转图像以便更符合日常习惯
            });
        }

        /***** 显示项目复选框 点击事件 ****/
        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            showBoxs = checkBox1.Checked;
        }
        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            showPoints = checkBox2.Checked;
        }
        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            showKeyPoint = checkBox3.Checked;
        }


        //////////////////
        /// 摄像机操作 ///
        //////////////////
        
        /***** 打开摄像头 ****/
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


        ////////////////////
        /// 图像处理操作 ///
        ////////////////////

        /***** 画关键点 ****/
        private Image<Bgr, byte> DrawKeyPoint(Image<Bgr, byte> img)
        {
            if (current_point.TTL >= 50)   //如果当前关键点的生存时间大于50 (表示它连续好多帧被识别为关键点)
            {                              //则绘制当前关键点 (蓝色)
                current_point.TTL -= 10;   //并将当前关键点的生存时间-10 (及时淘汰老旧的关键点, 提高精确度)
                img.Draw(new CircleF(new PointF((float)current_point.X, (float)current_point.Y), 5), new Bgr(255, 255, 0), 3);  //绘制当前关键点
                label1.Text = String.Format("{0, 3}, {1, 3}, {2, 3}", current_point.TTL, (int)current_point.X, (int)current_point.Y);
            } else
            {
                label1.Text = "No Key Point";
            }
            return img;
        }

        /***** 画检测点 ****/
        private Image<Bgr, byte> DrawPoints(Image<Bgr, byte> img)
        {
            listBox1.Items.Clear(); //清空列表框
            listBox1.Items.Add("生存时间, 横坐标, 纵坐标"); //清空列表框
            foreach (MyPoint point in points)  //遍历points数组
            {
                listBox1.Items.Add(String.Format("{0, 8}, {1, 6}, {2, 6}", point.TTL, (int)point.X, (int)point.Y));  //添加每个point到列表框
                if (point.TTL >= 50)  //如果此point的生存时间大于50
                {                     //则把它画出来
                    img.Draw(new CircleF(new PointF((float)point.X, (float)point.Y), 5), new Bgr(0, double.MaxValue, 0), 3);  //绘制此point
                }
            }
            return img;
        }
        
        /***** 检测手部 并画框 ****/
        private Image<Bgr, byte> ProcessImageUseDnn(Image<Bgr, byte> img)
        {
            Mat frame = img.Mat; //获取输入图像的矩阵

            Mat inputBlob = DnnInvoke.BlobFromImage(frame, 1, new Size(300, 300)); //转换为神经网络的输入格式 (300x300)

            net.SetInput(inputBlob, "image_tensor"); //输入数据
            Mat prob = net.Forward("detection_out"); //获取输出

            byte[] data = new byte[5600];  //将输出从矩阵转换为数组, 便于处理
            prob.CopyTo(data);

            for (int i = 0; i < prob.SizeOfDimemsion[2]; i++) //遍历检测框
            {
                var d = BitConverter.ToSingle(data, i * 28 + 8);  //获取检测框置信度
                if (d > 0.3) //如果置信度大于30%, 则认为有效
                {
                    var idx = (int)BitConverter.ToSingle(data, i * 28 + 4); //输出的类别, 这里只有hand一个类别
                    var w1 = (int)(BitConverter.ToSingle(data, i * 28 + 12) * img.Width);   //检测框的位置 - 左侧
                    var h1 = (int)(BitConverter.ToSingle(data, i * 28 + 16) * img.Height);  //检测框的位置 - 顶部
                    var w2 = (int)(BitConverter.ToSingle(data, i * 28 + 20) * img.Width);   //检测框的位置 - 右侧
                    var h2 = (int)(BitConverter.ToSingle(data, i * 28 + 24) * img.Height);  //检测框的位置 - 底部

                    double X = (w1 + w2) / 2; //检测框的位置 - 中心横坐标
                    double Y = (h1 + h2) / 2; //检测框的位置 - 中心纵坐标

                    AddPoint(X, Y); //调用添加检测点方法

                    if (!showBoxs)  //如果没有开启检测框绘制功能, 则跳过绘制检测框部分
                    {
                        continue;
                    }

                    var label = $"hand {d * 100:0.00}%";   //标签文本: "类别(hand) + 置信度"

                    CvInvoke.Rectangle(img, new Rectangle(w1, h1, w2 - w1, h2 - h1), new MCvScalar(0, 255, 0), 2); //新建一个矩形对象来表示检测框

                    int baseline = 0;
                    var textSize = CvInvoke.GetTextSize(label, FontFace.HersheyTriplex, 0.5, 1, ref baseline);     //计算标签大小
                    var y = h1 - textSize.Height < 0 ? h1 + textSize.Height : h1; //计算标签位置
                    CvInvoke.Rectangle(img, new Rectangle(w1, y - textSize.Height, textSize.Width, textSize.Height), new MCvScalar(0, 255, 0), -1);  //绘制检测框
                    CvInvoke.PutText(img, label, new Point(w1, y), FontFace.HersheyTriplex, 0.5, new Bgr(0, 0, 0).MCvScalar); //绘制标签
                }
            }

            UpdateTTL(); //调用更新生存时间方法

            return img;
        }


        //////////////////////
        /// 检测点处理操作 ///
        //////////////////////

        private List<MyPoint> points = new List<MyPoint>();
        private MyPoint current_point = new MyPoint(0, 0);

        /***** 更新(减小)每个检测点的生存时间, 并找出此帧的关键点 ****/
        public void UpdateTTL()
        {
            MyPoint max_point = null;
            for (int i = points.Count - 1; i >= 0; i--) //倒序遍历points数组
            {
                MyPoint point = points[i];
                point.TTL -= 10; //将每个检测点的生存时间减10
                if (!point.IsAlive) // 生存时间 < 0 时
                {
                    points.Remove(point); //删除检测点
                    continue;
                }
                if (current_point.TTL < 50) //当前没有关键点时
                {                           //直接把此点当作关键点
                    current_point.TTL = 90;
                    current_point.X = point.X;
                    current_point.Y = point.Y;
                    continue;
                }
                if (!IsOnePoint(current_point.X, current_point.Y, point.X, point.Y, 50)) //当前有关键点时,
                {                                                                        //判断此点距离关键点是否太远,
                    continue;                                                            //太远则肯定不会是下一个关键点
                }
                if (max_point == null) //当前没有最佳候选关键点时
                {                      //直接把此点当作最佳候选关键点
                    max_point = point;
                }
                else
                {
                    if (point.TTL > max_point.TTL) //取 此点 与 最佳候选关键点 中生存时间最大的为下一个最佳候选关键点
                    {
                        max_point = point;
                    }
                    else if (point.TTL == max_point.TTL) //此点 与 最佳候选关键点 生存时间相同时
                    {                                    //取 (此点) 与 (最佳候选关键点) 中距离 (当前关键点) 最近的的为下一个最佳候选关键点
                        double s_point = Math.Abs(current_point.X - point.X) + Math.Abs(current_point.Y - point.Y);
                        double s_max_point = Math.Abs(current_point.X - max_point.X) + Math.Abs(current_point.Y - max_point.Y);
                        if (s_max_point < s_point)
                        {
                            max_point = point;
                        }
                    }
                }
            }
            if (max_point != null)            //如果找到了本帧的 最佳候选关键点
            {                                 //则把它作为 当前关键点
                current_point.TTL += 20;      //并把它的生存时间加20
                current_point.X = max_point.X;
                current_point.Y = max_point.Y;
            }
        }

        /***** 添加检测点 ****/
        public void AddPoint(double x, double y)
        {
            bool isAdded = false;
            foreach (MyPoint point in points) //与当前points数组里的点逐个比较
            {
                if (IsOnePoint(x, y, point.X, point.Y)) //如果距离足够近
                {                                       //则当作同一个点, 不再添加新的点
                    point.TTL += 20;                    //并给此point的生存时间加20
                    point.X = (x + point.X) / 2;
                    point.Y = (y + point.Y) / 2;
                    isAdded = true;
                    break;
                }
            }
            if (!isAdded)                           //如果points数组中没有任何点距离新的点足够近
            {                                       //则把它当新的点, 添加进points数组
                MyPoint point = new MyPoint(x, y);  //并将此新的点的生存时间设为30 (并不会立即显示, 大于50才会显示)
                point.TTL = 30;
                points.Add(point);
            }
        }

        /***** 判断两点距离是否过近, 可以被看作为一个点 ****/
        public bool IsOnePoint(double x1, double y1, double x2, double y2, double size = 50)
        {
            return x1 < x2 + size && x1 > x2 - size &&
                y1 < y2 + size && y1 > y2 - size;
        }

    }


    ///////////////////////
    /// 自定义MyPoint类 ///
    ///////////////////////
    
    public class MyPoint
    {
        public double X;
        public double Y;
        double ttl;
        public double TTL
        {
            get { return ttl; }
            set
            {
                if (value <= 100)
                {
                    ttl = value;
                }
                else
                {
                    ttl = 100;
                }
            }
        }
        public bool IsAlive
        {
            get { return ttl > 0; }
        }

        public MyPoint(double x, double y)
        {
            this.X = x;
            this.Y = y;
        }
    }
}
