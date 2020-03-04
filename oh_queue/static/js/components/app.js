/* This component holds the global application state, and manages the websocket
 * connection. To update the state, call a method on the global "app" object,
 * e.g. as
 *
 *     app.addMessage("Something bad happened", "danger");
 *
 * Because it sits at the root of React heirarchy, any state changes in the app
 * will cause the entire app to re-render, so any state changes are reflected
 * instantly.
 *
 * All other React components are "stateless". Many of them are simply pure
 * functions that take the state and produce HTML. A few are slightly more
 * complicated in that they have to interact with jQuery or the network.
 *
 * NB: calling app methods inside a render() method will result in an infinite
 * loop.
 */
class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = initialState;

    let socket = connectSocket();
    this.socket = socket;
    socket.on('connect', () => {
      this.setOffline(false);
      this.refreshTickets();
    });
    socket.on('disconnect', () => this.setOffline(true));
    socket.on('state', (data) => this.updateState(data));
    socket.on('event', (data) => this.updateTicket(data));
    socket.on('presence', (data) => this.updatePresence(data));

    this.loadTicket = this.loadTicket.bind(this);
  }

  refresh() {
    this.setState(this.state);
  }

  setOffline(offline) {
    this.state.offline = offline;
    this.refresh();
  }

  updateState(data) {
    if (Array.isArray(data.assignments)) {
      this.state.assignments = {};
      for (var assignment of data.assignments) {
        this.state.assignments[assignment.id] = assignment;
      }
    }
    if (Array.isArray(data.locations)) {
      this.state.locations = {};
      for (var location of data.locations) {
        this.state.locations[location.id] = location;
      }
    }
    if (Array.isArray(data.tickets)) {
      for (var ticket of data.tickets) {
        setTicket(this.state, ticket);
      }
    }
    if (data.hasOwnProperty('config')) {
      this.state.config = data.config;
    }
    if (data.hasOwnProperty('current_user')) {
      this.state.currentUser = data.current_user;
    }
    if (data.hasOwnProperty('appointments')) {
        this.state.appointments = {};
        for (const appointment of data.appointments) {
            this.state.appointments[appointment.id] = appointment;
        }
    }
    this.state.loaded = true;
    this.refresh();
  }

  updatePresence(data) {
    this.state.presence = data;
    this.refresh();
  }

  refreshTickets() {
    let ticketIDs = Array.from(this.state.tickets.keys());
    this.socket.emit('refresh', ticketIDs, (data) => {
      for (var ticket of data.tickets) {
        setTicket(this.state, ticket);
      }
      this.refresh();
    });
  }

  shouldNotify(ticket, type) {
    return (isStaff(this.state) && type === 'create');
  }

  updateTicket(data) {
    setTicket(this.state, data.ticket);
    this.refresh();

    var ticket = data.ticket;
    switch(data.type) {
      case 'assign':
      case 'delete':
      case 'resolve':
        if (isStaff(this.state)) {
          cancelNotification(ticket.id + ".create");
        }
        break;
      case 'create':
      case 'describe':
      case 'unassign':
      case 'update_location':
        if (isStaff(this.state) && ticket.status === 'pending') {
          var assignment = ticketAssignment(this.state, ticket);
          var location = ticketLocation(this.state, ticket);
          var question = ticketQuestion(this.state, ticket);
          notifyUser('New request for ' + assignment.name + ' ' + question,
                     location.name,
                     ticket.id + '.create');
        }
        break;
    }
  }

  loadTicket(id) {
    loadTicket(this.state, id);
    this.refresh();
    this.socket.emit('load_ticket', id, (ticket) => {
      receiveTicket(this.state, id, ticket);
      this.refresh();
    });
  }

  toggleFilter() {
    this.state.filter.enabled = !this.state.filter.enabled;
    this.refresh();
  }

  setFilter(filter) {
    filter.enabled = !!this.state.filter.enabled;
    this.state.filter = filter;
    this.refresh();
  }

  addMessage(message, category) {
    addMessage(this.state, message, category);
    this.refresh();
  }

  clearMessage(id) {
    clearMessage(this.state, id);
    this.refresh();
  }

  makeRequest(eventType, request, follow_redirect=false, callback) {
    if (typeof request === "function") {
      follow_redirect = request;
      request = undefined;
    }
    if (typeof follow_redirect === "function") {
      callback = follow_redirect;
      follow_redirect = false;
    }
    let cb = (response) => {
      if (response == null) {
        if (callback) callback(response);
        return;
      }
      let messages = response.messages || [];
      for (var message of messages) {
        this.addMessage(message.text, message.category);
      }
      if (follow_redirect && response.redirect) {
        this.router.history.push(response.redirect);
      }
      if (callback) callback(response);
    };
    if (request !== undefined) {
      this.socket.emit(eventType, request, cb);
    } else {
      this.socket.emit(eventType, cb);
    }
  }

  render() {
    let { BrowserRouter, Route, Switch } = ReactRouterDOM;
    let state = this.state;
    return (
      <BrowserRouter ref={(router) => this.router = router}>
        <div>
          <Switch>
            <Route exact path="/" render={(props) => (<Home state={state} {...props} />)} />
            <Route path="/appointments" render={(props) => (<Appointments state={state} {...props} />)} />
            <Route path="/admin" render={(props) => (<AdminLayout state={state} {...props} />)} />
            <Route path="/error" render={(props) => (<ErrorView state={state} {...props} />)} />
            <Route path="/presence" render={(props) => (<PresenceIndicator state={state} {...props} />)} />
            <Route path="/tickets/:id" render={(props) => (<TicketLayout state={state} loadTicket={this.loadTicket} {...props} />)} />
            <Route render={(props) => (<ErrorView state={state} {...props} message="Page Not Found" />)} />
          </Switch>
        </div>
      </BrowserRouter>
    );
  }
}
