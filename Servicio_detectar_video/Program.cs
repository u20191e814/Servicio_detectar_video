using Servicio_detectar_video;
using Topshelf;

var exitcode = HostFactory.Run(x =>
{
    x.Service<Servicio>(s =>
    {
        s.ConstructUsing(ser => new Servicio());
        s.WhenStarted(ser => ser.Start());
        s.WhenStopped(ser => ser.Stop());

    });
    x.RunAsLocalService();
    x.SetServiceName("Upc_servicio_detectar_videos");
    x.SetDisplayName("Upc_servicio_detectar_videos");
    x.SetDescription("Upc_servicio_detectar_videos Lee los registros de base de datos y analiza los objetos");

});

var exitcodevalue = (int)Convert.ChangeType(exitcode, exitcode.GetTypeCode());
Environment.ExitCode = (exitcodevalue);