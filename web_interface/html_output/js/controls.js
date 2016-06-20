var video;
var controls;
var playControl;
var progressControl;
var progressHolder;
var playProgressBar;
var playProgressInterval;
var currentTimeDisplay;
var durationDisplay;
var volumeControl;
var volumeDisplay;
var fullScreenControl;

var videoWasPlaying;
var videoIsFullScreen;
var videoOrigWidth;
var videoOrigHeight;

var bodyLoaded = function(){
  video = document.getElementById("video");
  controls = document.getElementById("controls");
  playControl = document.getElementById("play");
  progressControl = document.getElementById("progress");
  progressHolder = document.getElementById("progress_box");
  playProgressBar = document.getElementById("play_progress");
  currentTimeDisplay = document.getElementById("current_time_display");
  durationDisplay = document.getElementById("duration_display");
  volumeControl = document.getElementById("volume");
  volumeDisplay = document.getElementById("volume_display");
  fullScreenControl = document.getElementById("full_screen");

  timelineControl = document.getElementById("timeline");
  timelineHolder = document.getElementById("progress_box2");
  timelineProgressBar = document.getElementById("play_progress2");
  controls2 = document.getElementById("controls2");

  showController();
  positionController();

  showController2();
  positionController2();

  trackPlayProgress();

  video.addEventListener("mouseup", function(){
    if (video.paused) {
      video.play();
      trackPlayProgress();
    } else {
      video.pause();
      stopTrackingPlayProgress();
    }
  }, true);   

  timelineHolder.addEventListener("mousedown", function(){
    stopTrackingPlayProgress();

    if (video.paused) {
      videoWasPlaying = false;
    } else {
      videoWasPlaying = true;
      video.pause();
    }

    blockTextSelection();
    document.onmousemove = function(e) {
      setPlayProgress2(e.pageX);
    }

    document.onmouseup = function() {
      unblockTextSelection();
      document.onmousemove = null;
      document.onmouseup = null;
      if (videoWasPlaying) {
        video.play();
        trackPlayProgress();
      }
    }
  }, true);

  timelineHolder.addEventListener("mouseup", function(e){
    setPlayProgress2(e.pageX);
  }, true);
}

function positionController(){
  controls.style.top = (video.offsetHeight - controls.offsetHeight) + "px";
  controls.style.left = "0px";
  controls.style.width = video.offsetWidth + "px";
}

function showController(){
  controls.style.display = "block";
}

function hideController(){
  controls.style.display = "none";
}

//timeline
function positionController2(){
  controls2.style.top = (video.offsetHeight - controls2.offsetHeight) + "px";
  controls2.style.left = "0px";
  controls2.style.width = video.offsetWidth + "px";
  sizeProgressBar2();
}

function showController2(){
  controls2.style.display = "block";
}

function sizeProgressBar2(){
  //timelineControl.style.width = (controls.offsetWidth - 125)*2 + "px";
  //timelineHolder.style.width = (progressControl.offsetWidth - 50)*4 + "px";
  timelineControl.style.width = 1853 + "px";
  timelineHolder.style.width = 1853 + "px";
  updatePlayProgress2();
}

function trackPlayProgress(){
  timelineProgressInterval = setInterval(updatePlayProgress2, 33);
}

function stopTrackingPlayProgress(){
  clearInterval(timelineProgressInterval);
}

function updatePlayProgress2(){
  timelineProgressBar.style.width = ((video.currentTime / video.duration) * (timelineHolder.offsetWidth - 2)) + "px";
  updateTimeDisplay();
}

function setPlayProgress2(clickX) {
  var newPercent = Math.max(0, Math.min(1, (clickX - findPosX(timelineHolder)) / timelineHolder.offsetWidth));
  video.currentTime = newPercent * video.duration
  timelineProgressBar.style.width = newPercent * (timelineHolder.offsetWidth)  + "px";
  updateTimeDisplay();
}



function updateTimeDisplay(){
  currentTimeDisplay.innerHTML = formatTime(video.currentTime);
  if (video.duration) durationDisplay.innerHTML = formatTime(video.duration);
}

function blockTextSelection(){
  document.body.focus();
  document.onselectstart = function () { return false; };
}

function unblockTextSelection(){
  document.onselectstart = function () { return true; };
}

// Return seconds as MM:SS
function formatTime(seconds) {
  seconds = Math.round(seconds);
  minutes = Math.floor(seconds / 60);
  minutes = (minutes >= 10) ? minutes : "0" + minutes;
  seconds = Math.floor(seconds % 60);
  seconds = (seconds >= 10) ? seconds : "0" + seconds;
  return minutes + ":" + seconds;
}

// Get objects position on the page
function findPosX(obj) {
  var curleft = obj.offsetLeft;
  while(obj = obj.offsetParent) {
    curleft += obj.offsetLeft;
  }
  return curleft;
}