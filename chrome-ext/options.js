document.addEventListener('DOMContentLoaded', function () {
  var settings = [
      {name: 'method'},
      {name: 'recommendNo'},
      {name: 'sortBy'}
  ];

  // Loop through the settings array and update the DOM elements and storage values
  settings.forEach(function(setting) {
      var element = document.getElementById(setting.name);
      console.log(element)
      chrome.storage.sync.get(setting.name, function (data) {
          element.value = data[setting.name];
      });
      element.addEventListener('change', function () {
          var value = element.value;
          var data = {};
          data[setting.name] = value;
          chrome.storage.sync.set(data);
      });
  });
});
