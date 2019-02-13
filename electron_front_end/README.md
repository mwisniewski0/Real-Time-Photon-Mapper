## Electron front end
This directory contains all of the code for the electron photon running files. the directory structure for this directory is mapped as follows

### Build and deployment
Build and deployment information can be found in the readme at the root of this project. Typical builds for development can be performed using 
basic npm commands.

to build use
```
npm start
```


```
photon-project-root/
|
...
|
|--> electron_front_end
|    |--> main.js # This js file contains the js code that is used
|    |            # as an entry point for the applicaiton
|    |--> index.html # This provides the styling and structure for
|    |               # main page that is rendered by the electron application
|    |--> package.json # project description for node package
|    |--> renderer.js # used for creation of main window page.
```