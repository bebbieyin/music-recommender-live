
 
// RECOMMENDATIONS PAGE JS

// play the songs in the mediaelementjs
     // Dynamic URL change
list.onclick = function(e) {
    /*e.preventDefault();*/
  
    var elm = e.target;
    var audio = document.getElementById('audio');
  
    var source = document.getElementById('audio');
    source.src = elm.getAttribute('data-value');
    var song_name = elm.getAttribute('data-name');
    var artist_name =  elm.getAttribute('data-artist');
    
    if (elm.getAttribute('data-value')) {
        audio.load(); //call this to just preload the audio without playing
        audio.play(); //call this to play the song right away 
        document.getElementById('nowplaying').innerHTML = song_name+' by '+artist_name

    }
  };


// SEARCH PAGES JS
function checkform()
{
  if (document.getElementById("myInput").value == "")
{
    // something is wrong
    alert('Please type in something!');
    return false;
}

return true;
}

// Logout of Spotify
function logout(){
    /* open a window that enables user to log out of spotify */
    var url = 'https://accounts.spotify.com/en/logout' 
    var spotifyLogoutWindow = window.open(url, 'Spotify Logout', 'width=400,height=110,top=40,left=40')
   
    /* close the window */ 
    setTimeout(function() { spotifyLogoutWindow.close();}, 1000);

    /*setTimeout(() => spotifyLogoutWindow.close(), 1000);  */

    }

// responsive navigation bar
$(function () { 

  $("#navbarToggle").blur(function (event) {
    var screenWidth = window.innerWidth;
    if (screenWidth < 768) {
      $("#collapsable-nav").collapse('hide');
    }
  });
});

// toggle like and dislike button on click
function toggleLike(x) {
  x.classList.toggle("fa-thumbs-up");
}
// toggle like and dislike button on click
function toggleDislike(x) {
  x.classList.toggle("fa-thumbs-down");
}

// PROFILE page - show hide recommendations
function showHide() {
  var div = document.getElementById('actions-button');
    div.style.display = 'block';

}


/*
        function changeSearchText(clicked_id){

            var search_type = clicked_id;
            var search_bar = document.getElementsByClassName("myclass");
        
            if (search_type == 'search-track') {
                search_bar.disabled=false;
            
            } else if (search_type=='search-artist'){
            
                search_bar.disabled=false;
                
            } else if (search_type=='search-genre'){
            
                search_bar.disabled=false;
                
            }

           
            $('input:text').attr('placeholder',placeholder_text);
           
        }
        */
