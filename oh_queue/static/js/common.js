function requestNotificationPermission() {
  if ('Notification' in window && Notification.permission !== 'denied') {
    Notification.requestPermission();
  }
}

function notifyUser(text, options) {
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(text, options);
  }
}

function connectSocket() {
  return io.connect('//' + document.domain + ':' + location.port, {
    transports: ['websocket', 'polling'],
  });
}

function goToTicket(nextTicketID) {
  let url = nextTicketID ? '/' + nextTicketID : '/';
  ReactRouter.browserHistory.push(url);
}

var socket = connectSocket();

ReactDOM.render(
  <ReactRouter.Router history={ReactRouter.browserHistory}>
    <ReactRouter.Route path="/" component={App}>
      <ReactRouter.IndexRoute component={Queue}/>
      <ReactRouter.Route path="/:id" component={TicketView}/>
    </ReactRouter.Route>
  </ReactRouter.Router>,

  document.getElementById('content')
);
