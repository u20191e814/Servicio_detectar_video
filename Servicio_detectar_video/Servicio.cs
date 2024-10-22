using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
using C_Servicio_detectar_imagenes;
using M_Servicio_detectar_imagenes;
using System.Drawing.Imaging;
using M_Servicio_detectar;
using static System.Net.Mime.MediaTypeNames;
using static Emgu.Util.Platform;

namespace Servicio_detectar_video
{
    public class Servicio
    {
        private readonly System.Timers.Timer _timer;


        private static EventViewer evento { get; set; } = null;
        private Configuracion confi { get; set; } = null;

        private static bool EstadoPlay { get; set; }


        public Servicio()
        {
            _timer = new System.Timers.Timer(1000)
            {
                AutoReset = false
            };
            _timer.Elapsed += Tiempo_Elapsed;


        }

        private int contadorVacio = 0;
        private void Tiempo_Elapsed(object sender, ElapsedEventArgs e)
        {
            _timer.Stop();

            try
            {
                List<MCvScalar> colores = new List<MCvScalar>();
                colores.Add(new MCvScalar(0, 0, 255));
                colores.Add(new MCvScalar(0, 255, 255));
                colores.Add(new MCvScalar(255, 0, 0));
                string[] classNames = File.ReadAllLines("obj.names");
                Net net = DnnInvoke.ReadNetFromDarknet("yolov4_Train_416.cfg", "yolov4_Train_final_416.weights");

                net.SetPreferableBackend(Emgu.CV.Dnn.Backend.Cuda);
                net.SetPreferableTarget(Target.Cuda);
                while (EstadoPlay)
                {
                    bool viewftp = false;
                    try
                    {

                        List<ObjectsInVideo> photos = ConsultasVideo.ObtenerRegistrosPendientes();
                        if (photos==null || photos.Count==0)
                        {
                            photos = ConsultasVideo.ObtenerRegistrosPendientesFolder();
                        }
                        if (photos != null && photos.Count() > 0)
                        {
                            contadorVacio = 0;
                            Message.Success("Se han encontrado " + photos.Count() + " a ser analizados");

                            foreach (var item in photos)
                            {
                                Etiqueta:

                                DateTime inicio = DateTime.Now;
                                if (!File.Exists(item.video_to_detect))
                                {
                                    Message.warning("No existe el archivo ");
                                    continue;
                                }
                                VideoCapture video = new VideoCapture(item.video_to_detect);
                                double fps= video.Get(CapProp.Fps);
                                double totalframes = video.Get(CapProp.FrameCount);
                                int duracion =(int)( totalframes / fps);
                                TimeSpan dur = new TimeSpan(0,0,duracion);
                                string t_dur = dur.ToString(@"hh\:mm\:ss");
                                bool estado = false;
                                if (item.PK_ObjectsInVideo>0)
                                {
                                    estado = ConsultasVideo.ActualizarVideo(t_dur, item.PK_ObjectsInVideo);
                                }
                                else
                                {
                                    estado = ConsultasVideo.ActualizarVideoFolder(t_dur, item.PK_ObjectInVideoFolder);
                                }
                                
                                if (!estado)
                                {
                                    Message.Error("No se pudo actualizar la duración ");
                                    goto Etiqueta;
                                }
                                int fourcc= Emgu.CV.VideoWriter.Fourcc('X','2','6','4');
                                int width =(int) video.Get(CapProp.FrameWidth);
                                int height = (int) video.Get(CapProp.FrameHeight);
                                string rutasalidaMp4 = item.video_to_detect.Replace(".mp4", "_result.mp4");
                                VideoWriter vwriter = new VideoWriter(rutasalidaMp4,fourcc, (int)fps, new Size(width, height), true);
                                if (item.PK_ObjectsInVideo>0)
                                {
                                    ConsultasVideo.EliminarRegistros(item.PK_ObjectsInVideo);
                                    ConsultasVideo.ActualizarEstado(item.PK_ObjectsInVideo, 2);
                                }
                                else
                                {
                                    ConsultasVideo.EliminarRegistrosFolder(item.PK_ObjectInVideoFolder);
                                    ConsultasVideo.ActualizarEstadoFolder(item.PK_ObjectInVideoFolder, 2);
                                }
                                Mat copiaImagen = new Mat();
                                bool Analisis = false;
                                int current_frame = 0;
                                int contadorImagendetectado = 0;
                                DateTime tini = DateTime.Now;
                                while (true)
                                {
                                    Mat frame = new Mat();
                                    bool lectura =video.Read(frame);
                                    if (lectura)
                                    {
                                        if ((current_frame % 10) == 0)
                                        {
                                            int segundo = (int)(current_frame/fps);
                                            Message.Info("Analizando "+ segundo + " seg");
                                            Mat blob = DnnInvoke.BlobFromImage(frame, 1 / 255.0, new Size(416, 416), new MCvScalar(0, 0, 0), true, false);
                                            VectorOfMat layerOutputs = new VectorOfMat();
                                            string[] outNames = net.UnconnectedOutLayersNames;

                                            net.SetInput(blob);
                                            net.Forward(layerOutputs, outNames);
                                            blob.Dispose();

                                            List<Rectangle> boxes = new List<Rectangle>();
                                            List<float> confidences = new List<float>();
                                            List<int> classIDs = new List<int>();
                                            float ConfidenceThreshold = item.score / 100f;

                                            for (int k = 0; k < layerOutputs.Size; k++)
                                            {
                                                float[,] lo = (float[,])layerOutputs[k].GetData();
                                                int len = lo.GetLength(0);
                                                for (int i = 0; i < len; i++)
                                                {
                                                    if (lo[i, 4] < ConfidenceThreshold)
                                                        continue;
                                                    float max = 0;
                                                    int idx = 0;

                                                    int len2 = lo.GetLength(1);
                                                    for (int j = 5; j < len2; j++)
                                                        if (lo[i, j] > max)
                                                        {
                                                            max = lo[i, j];
                                                            idx = j - 5;
                                                        }

                                                    if (max > ConfidenceThreshold)
                                                    {
                                                        lo[i, 0] *= width;
                                                        lo[i, 1] *= height;
                                                        lo[i, 2] *= width;
                                                        lo[i, 3] *= height;

                                                        int x = (int)(lo[i, 0] - (lo[i, 2] / 2));
                                                        int y = (int)(lo[i, 1] - (lo[i, 3] / 2));

                                                        var rect = new Rectangle(x, y, (int)lo[i, 2], (int)lo[i, 3]);

                                                        rect.X = rect.X < 0 ? 0 : rect.X;
                                                        rect.X = rect.X > width ? width - 1 : rect.X;
                                                        rect.Y = rect.Y < 0 ? 0 : rect.Y;
                                                        rect.Y = rect.Y > height ? height - 1 : rect.Y;
                                                        rect.Width = rect.X + rect.Width > width ? width - rect.X - 1 : rect.Width;
                                                        rect.Height = rect.Y + rect.Height > height ? height - rect.Y - 1 : rect.Height;

                                                        boxes.Add(rect);
                                                        confidences.Add(max);
                                                        classIDs.Add(idx);
                                                    }
                                                }
                                            }

                                            int[] bIndexes = DnnInvoke.NMSBoxes(boxes.ToArray(), confidences.ToArray(), ConfidenceThreshold, ConfidenceThreshold);

                                            if (bIndexes.Length > 0)
                                            {

                                                Analisis = true;
                                                contadorImagendetectado++;
                                                foreach (var idx in bIndexes)
                                                {
                                                    var rc = boxes[idx];
                                                    double ccn = Math.Round(confidences[idx], 4);
                                                    int id = classIDs[idx];
                                                    string clase = classNames[classIDs[idx]];
                                                    var color = colores[id];
                                                    CvInvoke.Rectangle(frame, boxes[idx], color, 2);
                                                    int score = (int)(confidences[idx] * 100);
                                                    string colrs = "rgb(" + color.V2 + "," + color.V1 + "," + color.V0 + ")";
                                                    if (item.PK_ObjectsInVideo > 0)
                                                    {
                                                        
                                                        List<dynamic> registros = ConsultasVideo.ValidarInfo(clase, (segundo + 1), item.PK_ObjectsInVideo);
                                                        bool insertar = true;
                                                        if (registros!= null && registros.Count>0)
                                                        {
                                                            foreach (dynamic rt in registros)
                                                            {
                                                                Rectangle rr = new Rectangle(rt.x, rt.y, rt.w, rt.h);
                                                                if (rr.IntersectsWith(rc))
                                                                {
                                                                   insertar = false;
                                                                    
                                                                }
                                                            }
                                                        }

                                                        if (insertar)
                                                        {
                                                            ConsultasVideo.InsertarObjeto(rc.X, rc.Y, rc.Width, rc.Height, score, clase, colrs, (segundo + 1), item.PK_ObjectsInVideo);
                                                        }
                                                    }
                                                    else
                                                    {
                                                        List<dynamic> registros = ConsultasVideo.ValidarInfoFolder(clase, (segundo + 1), item.PK_ObjectInVideoFolder);
                                                        bool insertar = true;
                                                        if (registros != null && registros.Count > 0)
                                                        {
                                                            foreach (dynamic rt in registros)
                                                            {
                                                                Rectangle rr = new Rectangle(rt.x, rt.y, rt.w, rt.h);
                                                                if (rr.IntersectsWith(rc))
                                                                {
                                                                    insertar = false;

                                                                }
                                                            }
                                                        }
                                                        if (insertar)
                                                        {
                                                            ConsultasVideo.InsertarObjetoFolder(rc.X, rc.Y, rc.Width, rc.Height, score, clase, colrs, (segundo + 1), item.PK_ObjectInVideoFolder);
                                                        }
                                                    }


                                                }
                                                copiaImagen = frame.Clone() as Mat;
                                            }
                                            else
                                            {
                                                Analisis = false;
                                            }
                                            layerOutputs.Dispose();


                                        }
                                        if (Analisis)
                                        {
                                           
                                            vwriter.Write(copiaImagen);
                                        }
                                        else
                                        {
                                            vwriter.Write(frame);
                                        }
                                        frame.Dispose();
                                        current_frame += 1;
                                    }
                                    else
                                    {
                                        break;
                                    }
                                }
                                DateTime tfin = DateTime.Now;
                                var vyh = tfin - tini;
                                
                                vwriter.Dispose();
                                video.Dispose();
                                copiaImagen.Dispose();
                                if (contadorImagendetectado > 0)
                                {
                                    if (item.PK_ObjectsInVideo>0)
                                    {
                                        ConsultasVideo.ActualizarResultado(rutasalidaMp4, item.PK_ObjectsInVideo);
                                    }
                                    else
                                    {
                                        ConsultasVideo.ActualizarResultadoFolder(rutasalidaMp4, item.PK_ObjectInVideoFolder);
                                    }
                                    
                                }
                                else
                                {
                                    File.Delete(rutasalidaMp4);
                                }
                                if (item.PK_ObjectsInVideo>0)
                                {
                                    ConsultasVideo.ActualizarEstado(item.PK_ObjectsInVideo, 1);
                                }
                                else
                                {
                                    ConsultasVideo.ActualizarEstadoFolder(item.PK_ObjectInVideoFolder, 1);
                                }
                                DateTime fin = DateTime.Now;
                                TimeSpan diff = fin - inicio;
                                Message.warning("Duración: " + diff.ToString(@"hh\:mm\:ss\.fff"));
                                Message.Success("FIN DEL PROCESO");
                            }
                        }
                        else
                        {
                            contadorVacio++;
                            if (contadorVacio > 20)
                            {
                                contadorVacio = 0;
                                viewftp = true;
                                Console.Clear();
                            }
                            Message.warning("No se existen imagenes para procesar");

                        }
                    }
                    catch (Exception ex)
                    {
                        Message.Error("While", ex);
                    }
                    if (viewftp)
                    {
                        Thread.Sleep(5000);
                    }
                    else
                    {
                        Thread.Sleep(100);
                    }

                }
            }
            catch (Exception ex)
            {

                Message.Error("Periodo tiempo", ex);
            }

        }


        private bool IniciarAplicacion()
        {
            bool activacion = false;
            try
            {

                activacion = true;
               
                Message.Success("Aplicación exitosa");

            }
            catch (Exception ex)
            {
                Message.Error("Error en Iniciar aplicacion  " + ex.Message + ". (Error: 108000070)");


            }
            return activacion;
        }




        public void Start()
        {

            evento = new EventViewer();

            evento.WriteErrorLog("Inicio de aplicación ");
            evento.WriteEventViewerLog("Inicio de aplicación ", 4);
            confi = evento.config;
            bool estado = IniciarAplicacion();
            if (estado)
            {
                _timer.Start();
                EstadoPlay = true;
            }


        }
        public void Stop()
        {
            EstadoPlay = false;
            _timer.Stop();
        }
    }
}
