# Photon - A Real Time Photon Mapper

Photon is a rendering engine designed for viewing models of 3D spaces in real time while maintaining realistic light simulation present in ray tracers and path tracers. This is achieved by precomputing a global illumination solution to a still scene (we are using photon mapping for this stage) and using that information to render the scene in real time. The project is aimed at architects, realtors, interior designers, etc. who want to show realistic 3D spaces to their customers and allow them to freely explore them.

Currently, Photon uses CUDA for both the precomputing stage (stage 1), and the real time rendering stage (stage 2). To achieve cross-platform support, we are using SDL2 for interacting with the user and acquiring an OpenGL context.

## How to build

Currently, Photon builds on macOS, Linux and Windows. Since it is a CUDA project you will need the CUDA compiler (nvcc) to build it. Moreover, due to our dependency on SDL2 and GLEW, you will also need versions of these libraries for your environment.

### Windows using Visual Studio

* Your Visual Studio installation will need to include the C++ desktop development components (note that these are not included in the default installation).
* Install CUDA for Visual Studio from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
* Clone this repository
* Open the [photon.sln](/photon.sln) solution file in Visual Studio
* You will need precompiled versions of SDL2 and GLEW; the project expects them to be in a `deps` directory. Simply download the [windows.zip file](https://github.com/mwisniewski0/Real-Time-Photon-Mapper/blob/deps/windows.zip) from the [dependencies branch](https://github.com/mwisniewski0/Real-Time-Photon-Mapper/tree/deps) and unpack it in the solution directory.
* The project will now build correctly

## Building the electron frontend

### Install node.js and npm

In order to develop and build the electron front end, you need to first begin by downloading node.js. Instructions for specific desktop environments can be found at [nodejs]https://nodejs.org/en/download/.

```bash
install electron
```
after, you must install the electron libraries for use in the project. This can be achieved using the command below.

```bash
npm install -g electron
```

### Building for Development
In order to run the electron application in the development environment, you must install all associated packages. Change your current working directory to electron_front_end/ then run the command.

```bash
npm install
```

This will download all necessary libraries and node packages required to run the front end interface into a directory titled node_modules. From this point on, you can run the electron front end in a development environment by calling the command blow.

```bash
npm start
```

### Deployment
we use the [electron packager]https://github.com/electron-userland/electron-packager node in order to build the front end application for deployment. To use, migrate to the electron_front_end directory and use the instructions listed on their repo to install and deploy.

## Hardware requirements

### Stage 1

* NVidia GPU with compute capability >= 2.0
* At least 6 GB of VRAM
* At least 8 GB of RAM

### Stage 2

* NVidia GPU with compute capability >= 2.0
* At least 2 GB of VRAM
* At least 8 GB of RAM

## Contributors' notes

* Please browse this project's open issues to find a specific piece of the software to work on.
* Each section of the program contains a more specific readme which explains the section in more detail.

## How to use Photon?

The general Photon workflow is divided into a few steps. First, an appropriate `.obj` file (together with its `.mtl` file and texture files) needs to be located. Such files can be created by most modern 3D-modeling software such as Blender, SketchUp or Autodesk 3DS Max.

Once you localize your `.obj` model, it is ready to be processed into a `.photon` file by the first stage of Photon. Localize your stage1 executable as built in the *How to build* section above. Stage1 is a command-line program with the following interface:

```
stage1 [FLAGS]

-i, --input arg   Path to the obj file. Note that .mtl files and texture
                  files are expected to be in the same directory
-o, --output arg  Path to the output photon file
    --help        Print help

```

For example, if one wants to create a `.photon` file from an `.obj` file located at: `C:\Users\user\model\model.obj` and save it to `C:\Users\user\photon\model.photon`, one should run:

```.\stage1.exe --input=C:\Users\user\model\model.obj --output=C:\Users\user\photon\model.photon```

Should the output flag be omitted, the `.photon` file will be written in the current working directory.

After producing your `.photon` file, it is ready for real time rendering using the stage2 of Photon. Stage2 has to be started through a command-line interface (or through the electron frontend described in sections above):

```
  stage2 [FLAGS]

  -w, --width arg   Width of the output window (default: 400)
  -h, --height arg  Height of the output window (default: 300)
  -f, --hfov arg    Horizontal field of view in degrees (default: 40.0)
  -i, --input arg   Path to the photon file
      --help        Print help
```

For example, if one was to render a `.photon` file located at `C:\Users\user\photon\model.photon`, one should run:

```.\stage2.exe -i C:\Users\user\photon\model.photon```

Afterwards, a real-time render window will open. The user can interact with it by:

* Moving the mouse to turn the camera
* Use WASD keys to walk around the scene: W - to move forward, A - to move left, S - to move backward, D - to move right

Note, that you can change the parameters of the render window using the flags explained above. For example, to start a window in 1080p with 55 degrees of horizontal field of view, one would run:

```.\stage2.exe -i C:\Users\user\photon\model.photon -w 1920 -h 1080 -f 55```
`