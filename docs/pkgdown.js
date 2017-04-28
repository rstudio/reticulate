$(function() {
  $("#sidebar").stick_in_parent({offset_top: 40});
  $('body').scrollspy({
    target: '#sidebar',
    offset: 60
  });

});
