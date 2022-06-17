using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Configuration;

namespace WinForms_ObjectDetection
{
    public partial class Form1 : Form
    {
        Dictionary<string, Image<Bgr, byte>> IMGDict;
        VideoCapture videoCapture = null;
        string[] CocoClasses;
        Net CaffeModel = null;
        public Form1()
        {
            InitializeComponent();
            IMGDict = new Dictionary<string, Image<Bgr, byte>>();
        }

        private void toolStripMenuItemOpen_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog = new OpenFileDialog();
                dialog.Filter = "Video Files(*.avi;*.mp4;)|*.avi;*.mp4;|All Files(*.*;)|*.*;";

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    if (videoCapture != null && videoCapture.IsOpened)
                    {
                        videoCapture.Dispose();
                        videoCapture = null;
                    }

                    videoCapture = new VideoCapture(dialog.FileName);
                    Mat frame = new Mat();
                    videoCapture.Read(frame);

                    pictureBox1.Image = frame.ToBitmap();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

        }

        private void processObjectDetection(object sender, EventArgs e)
        {
            try
            {
                Mat frame = new Mat();
                videoCapture.Read(frame);
                if (frame == null)
                {
                    return;
                }

                var img = frame.ToImage<Bgr, byte>();
                var blob = DnnInvoke.BlobFromImage(img, 1.0, frame.Size, swapRB: true);
                CaffeModel.SetInput(blob);

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

                        // only person
                        //if (classID!=0)
                        //{
                        //    continue;
                        //}

                        int left = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 3) * img.Cols);
                        int top = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 4) * img.Rows);
                        int right = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 5) * img.Cols);
                        int bottom = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 6) * img.Rows);

                        Rectangle rectangle = new Rectangle(left, top, right - left + 1, bottom - top + 1);
                        img.Draw(rectangle, new Bgr(0, 0, 255), 2);
                        var labels = CocoClasses[classID];
                        CvInvoke.PutText(img, labels, new Point(left, top - 10), FontFace.HersheySimplex, 1.0,
                            new MCvScalar(0, 255, 0), 2);

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

        private void startRCNNToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (videoCapture == null)
                {
                    throw new Exception("Please load a video");
                }

                var modelpath = ConfigurationManager.AppSettings["ModelPath"];
                var configPath = ConfigurationManager.AppSettings["ModelArchitecture"];
                var coconamespath = ConfigurationManager.AppSettings["CocoClasses"];

                CaffeModel = DnnInvoke.ReadNetFromTensorflow(modelpath, configPath);
                CaffeModel.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                CaffeModel.SetPreferableTarget(Target.Cpu);

                CocoClasses = File.ReadAllLines(coconamespath);

                videoCapture.ImageGrabbed += processObjectDetection;
                videoCapture.Start();


            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void findContureToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                Mat frame = new Mat();
                videoCapture.Read(frame);
                if (frame == null)
                {
                    return;
                }

                var videoToImgInRgb = frame.ToImage<Bgr, byte>();
                //var hsv = imgBgr.ToBitmap<Hsv, byte>();

                //Izolacija samo bijelih linija
                var maskWhiteLines = videoToImgInRgb.InRange(new Bgr(190, 190, 190), new Bgr(255, 255, 255));
                pictureBox2.Image = maskWhiteLines.ToBitmap();
                //imgEdgeDetection = imgBgr.Canny(300, 50);
                var imgWithLines = new Image<Bgr, byte>(maskWhiteLines.Width, maskWhiteLines.Height, new Bgr());

                //pictureBox1.Image = imgEdgeDetection.ToBitmap();

                //Trazenje pravih linija
                var lines = maskWhiteLines.HoughLinesBinary(2, Math.PI / 100, 50, 3, 10);

                //izolacija potrebnih linija, potrebne duzine
                for (int i = 0; i < lines[0].Length; i++)
                {
                    //if (lines[0][i].Length > 10 && lines[0][i].Length < 30)
                    //{
                    //    //imgEdgeDetection.Draw(lines[0][i], new Bgr(0, 200, 0), 3);
                    //    //green
                    //    img2.Draw(lines[0][i], new Bgr(0, 255, 0), 3);
                    //}
                    //if (lines[0][i].Length > 30 && lines[0][i].Length < 60)
                    //{
                    //    //blue
                    //    img2.Draw(lines[0][i], new Bgr(255, 0, 0), 3);
                    //}
                    if (lines[0][i].Length > 60)
                    {
                        //red
                        imgWithLines.Draw(lines[0][i], new Bgr(0, 0, 255), 3);
                    }

                }
                pictureBox2.Image = imgWithLines.ToBitmap();

                Image<Gray, byte> imgEdgeDetection = imgWithLines.Canny(500, 50);

                pictureBox2.Image = imgEdgeDetection.ToBitmap();

                FindBoundingBoxes(imgEdgeDetection.ToBitmap());

                //Izolacija sa detekcijom ivica


                //var imgEdgeDetection = new Image<Gray, byte>(imgBgr.Width, imgBgr.Height, new Gray(0));

                //imgEdgeDetection = imgBgr.Canny(300, 50);
                //var img2 = new Image<Bgr, byte>(imgEdgeDetection.Width, imgEdgeDetection.Height, new Bgr());

                //pictureBox1.Image = imgEdgeDetection.ToBitmap();

                //var lines = imgEdgeDetection.HoughLinesBinary(2, Math.PI / 100, 50, 3, 10);

                //for (int i = 0; i < lines[0].Length; i++)
                //{
                //    if (lines[0][i].Length > 10 && lines[0][i].Length < 30)
                //    {
                //        //imgEdgeDetection.Draw(lines[0][i], new Bgr(0, 200, 0), 3);
                //        //green
                //        img2.Draw(lines[0][i], new Bgr(0, 255, 0), 3);
                //    }
                //    if (lines[0][i].Length > 30 && lines[0][i].Length < 60)
                //    {
                //        //blue
                //        img2.Draw(lines[0][i], new Bgr(255, 0, 0), 3);
                //    }
                //    if (lines[0][i].Length > 60 )
                //    {
                //        //red
                //        img2.Draw(lines[0][i], new Bgr(0, 0, 255), 3);
                //    }

                //}
                //pictureBox2.Image = img2.ToBitmap();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }


        }

        private void FindBoundingBoxes(Image image)
        {

        }

        private void toolStripMenuItemExit_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void findROIToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {

                Mat frame = new Mat();
                videoCapture.Read(frame);
                if (frame == null)
                {
                    return;
                }

                var img = frame.ToImage<Bgr, byte>();


                Rectangle A1 = new Rectangle(130, 98, 70, 100);
                Rectangle A2 = new Rectangle(210, 98, 70, 100);
                Rectangle A3 = new Rectangle(290, 98, 70, 100);

                Rectangle A4 = new Rectangle(110, 260, 70, 110);
                Rectangle A5 = new Rectangle(190, 260, 70, 110);
                Rectangle A6 = new Rectangle(270, 260, 70, 110);



                Rectangle B1 = new Rectangle(530, 110, 70, 100);
                Rectangle B2 = new Rectangle(610, 110, 70, 100);
                Rectangle B3 = new Rectangle(690, 110, 70, 100);

                Rectangle B4 = new Rectangle(540, 260, 70, 110);
                Rectangle B5 = new Rectangle(620, 260, 70, 110);
                Rectangle B6 = new Rectangle(700, 260, 70, 110);



                img.Draw(A1, new Bgr(0, 0, 255), 2);
                img.Draw(A2, new Bgr(0, 0, 255), 2);
                img.Draw(A3, new Bgr(0, 0, 255), 2);
                img.Draw(A4, new Bgr(0, 0, 255), 2);
                img.Draw(A5, new Bgr(0, 0, 255), 2);
                img.Draw(A6, new Bgr(0, 0, 255), 2);

                img.Draw(B1, new Bgr(0, 0, 255), 2);
                img.Draw(B2, new Bgr(0, 0, 255), 2);
                img.Draw(B3, new Bgr(0, 0, 255), 2);
                img.Draw(B4, new Bgr(0, 0, 255), 2);
                img.Draw(B5, new Bgr(0, 0, 255), 2);
                img.Draw(B6, new Bgr(0, 0, 255), 2);



                pictureBox2.Image = img.ToBitmap();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void processObjDetInROI(object sender, EventArgs e)
        {
            try
            {
                Mat frame = new Mat();
                videoCapture.Read(frame);
                if (frame == null)
                {
                    return;
                }

                var img = frame.ToImage<Bgr, byte>();

                Rectangle A1 = new Rectangle(130, 98, 70, 100);
                Rectangle A2 = new Rectangle(210, 98, 70, 100);
                Rectangle A3 = new Rectangle(290, 98, 70, 100);

                Rectangle A4 = new Rectangle(110, 260, 70, 110);
                Rectangle A5 = new Rectangle(190, 260, 70, 110);
                Rectangle A6 = new Rectangle(270, 260, 70, 110);



                //Rectangle B1 = new Rectangle(530, 110, 70, 100);
                //Rectangle B2 = new Rectangle(610, 110, 70, 100);
                //Rectangle B3 = new Rectangle(690, 110, 70, 100);

                //Rectangle B4 = new Rectangle(540, 260, 70, 110);
                //Rectangle B5 = new Rectangle(620, 260, 70, 110);
                //Rectangle B6 = new Rectangle(700, 260, 70, 110);



                img.Draw(A1, new Bgr(0, 0, 255), 2);
                img.Draw(A2, new Bgr(0, 0, 255), 2);
                img.Draw(A3, new Bgr(0, 0, 255), 2);
                img.Draw(A4, new Bgr(0, 0, 255), 2);
                img.Draw(A5, new Bgr(0, 0, 255), 2);
                img.Draw(A6, new Bgr(0, 0, 255), 2);

                //img.Draw(B1, new Bgr(0, 0, 255), 2);
                //img.Draw(B2, new Bgr(0, 0, 255), 2);
                //img.Draw(B3, new Bgr(0, 0, 255), 2);
                //img.Draw(B4, new Bgr(0, 0, 255), 2);
                //img.Draw(B5, new Bgr(0, 0, 255), 2);
                //img.Draw(B6, new Bgr(0, 0, 255), 2);


                var blob = DnnInvoke.BlobFromImage(img, 1.0, A1.Size, swapRB: true);
                CaffeModel.SetInput(blob);

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

                        // only person
                        if (classID != 2)
                        {
                            continue;
                        }

                        //int left = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 3) * img.Cols);
                        //int top = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 4) * img.Rows);
                        //int right = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 5) * img.Cols);
                        //int bottom = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 6) * img.Rows);

                        //Rectangle rectangle = new Rectangle(left, top, right - left + 1, bottom - top + 1);
                        //img.Draw(rectangle, new Bgr(0, 0, 255), 2);
                        //var labels = CocoClasses[classID];
                        //CvInvoke.PutText(img, labels, new Point(left, top - 10), FontFace.HersheySimplex, 1.0,
                        //    new MCvScalar(0, 255, 0), 2);
                        img.Draw(A1, new Bgr(0, 255, 0), 2);
                        var labels = CocoClasses[classID];
                        CvInvoke.PutText(img, labels, new Point(530, 503 - 10), FontFace.HersheySimplex, 1.0,
                            new MCvScalar(0, 255, 0), 2);

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

    }
}