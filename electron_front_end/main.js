// Modules to control application life and create native browser window
const electron = require('electron');

const {app, BrowserWindow, Menu, dialog, ipcMain} = electron;
const cp = require("child_process");

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow () {
  mainWindow = new BrowserWindow({width: 800, height: 600})
  mainWindow.loadFile('index.html')
  mainWindow.on('closed', function () {
    mainWindow = null
  })
}


ipcMain.on('file:find', (e, item) => {
    const files = dialog.showOpenDialog(mainWindow, {properties: ['openFile']});
    if (files[0]) {
        if (files[0].slice(-7) == ".photon") {
            if (process.platform == 'darwin') {
                cp.exec("open " + files[0]);
            }
            else if (process.platform == "win32") {
                cp.exec("open " + files[0]);
            }
            else {
                cp.exec("xdg-open " + files[0]);
            }
            e.sender.send('file:valid', files[0]);

        }
        else {
            e.sender.send("file:invalid", files[0]);
        }
    }
});

// This method will be called when Electron has finished initialization
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', function () {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
    createWindow()
  }
})
