using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.Data.SqlClient;
using System.Configuration;

namespace WinForms_ObjectDetection
{
    public partial class Form1 : Form
    {
        SqlConnection con = new SqlConnection(ConfigurationManager.AppSettings["ConnString"]);
        SqlCommand cmd;
        string[] CocoClasses;
        Net CaffeModel = null;
        VideoCapture videoCapture;

        public Form1()
        {
            InitializeComponent();
        }

        private void processObjectDetection(object sender, EventArgs e)
        {
            try
            {
                Rectangle A1 = new Rectangle(80, 120, 190, 150);
                Rectangle A2 = new Rectangle(250, 150, 190, 190);
                Rectangle A3 = new Rectangle(420, 200, 190, 180);

                Rectangle A4 = new Rectangle(280, 40,110, 70);
                Rectangle A5 = new Rectangle(400, 70, 110, 80);
                Rectangle A6 = new Rectangle(520, 80, 110, 100);

                var rois = new List<Rectangle>(); // List of ROIs
                rois.Add(A1);
                rois.Add(A2);
                rois.Add(A3); 
                rois.Add(A4);
                rois.Add(A5);
                rois.Add(A6);

                Mat frame = new Mat();
                videoCapture.Retrieve(frame);

                var img = frame.ToImage<Bgr, byte>();
                var blob = DnnInvoke.BlobFromImage(img, 1.0, frame.Size, swapRB: true);
                CaffeModel.SetInput(blob);

                foreach (var roi in rois)
                {
                   img.Draw(roi, new Bgr(0, 255, 0), 2);
                }

                var output = new VectorOfMat();
                string[] outnames = new string[] { "detection_out_final" };
                CaffeModel.Forward(output, outnames);

                var threshold = 0.6;
                int numDetections = output[0].SizeOfDimension[2];
                int numClasses = 90;

                var bboxes = output[0].GetData();

                for (int i = 0; i < numDetections; i++)
                {
                    float score = (float)bboxes.GetValue(0, 0, i, 2);
                    if (score > threshold)
                    {
                        int classID = Convert.ToInt32(bboxes.GetValue(0, 0, i, 1));

                        //only car
                        //if (classID != 3)
                        //{
                        //    continue;
                        //}

                        int left = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 3) * img.Cols);
                        int top = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 4) * img.Rows);
                        int right = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 5) * img.Cols);
                        int bottom = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 6) * img.Rows);

                        Rectangle rectangle = new Rectangle(left, top, right - left + 1, bottom - top + 1);
                        Point WeightedCentroid = new Point((rectangle.Left + rectangle.Right) / 2, (rectangle.Top + rectangle.Bottom) / 2);

                        img.Draw(rectangle, new Bgr(255, 0, 0), 2);
                        CvInvoke.Circle(img, WeightedCentroid, 4, new MCvScalar(0, 255, 0), 5, LineType.EightConnected, 0);
                        Point WeightedCentroidnew = WeightedCentroid;
                         CvInvoke.Line(img, WeightedCentroid, WeightedCentroidnew, new MCvScalar(255, 0, 0), 5, LineType.EightConnected, 0);
                        
                        foreach(var (roi, n) in rois.Select((roi, n) => (roi, n)))
                        {
                            if (roi.Contains(WeightedCentroid))
                            {
                                img.Draw(roi, new Bgr(0, 0, 255), 2);
                                updateDatabase("A" + (n+1));
                            }

                        }

                        var labels = CocoClasses[classID];
                        CvInvoke.PutText(img, labels, new Point(left, top - 10), FontFace.HersheySimplex, 1.0,
                            new MCvScalar(0, 0, 255), 2);
                    }
                }

                pictureBox1.Invoke((MethodInvoker)delegate
                {
                    pictureBox1.Image = img.ToBitmap();

                });
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }

        public void updateDatabase(string oznaka)
        {
            try
            {
                cmd = new SqlCommand("update Mjesto set Zauzeto=@zauzeto where Oznaka=@oznaka", con);
                con.Open();
                cmd.Parameters.AddWithValue("@zauzeto", 1);
                cmd.Parameters.AddWithValue("@oznaka", oznaka);
                cmd.ExecuteNonQuery();
                con.Close();

            }
            catch (Exception)
            {
                MessageBox.Show("Record update fail!");
            }
        }

        private void startMaskRCNN()
        {
            try
            {
                var modelpath = ConfigurationManager.AppSettings["ModelPath"];
                var configPath = ConfigurationManager.AppSettings["ModelArchitecture"];
                var coconamespath = ConfigurationManager.AppSettings["CocoClasses"];

                CaffeModel = DnnInvoke.ReadNetFromTensorflow(modelpath, configPath);
                CaffeModel.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                CaffeModel.SetPreferableTarget(Target.Cpu);

                CocoClasses = File.ReadAllLines(coconamespath);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (videoCapture == null)
            {
                videoCapture = new VideoCapture(0);
                startMaskRCNN();
                videoCapture.ImageGrabbed += processObjectDetection;
                videoCapture.Start();
            }
        }
    }
}