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
