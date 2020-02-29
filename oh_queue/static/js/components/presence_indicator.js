let PresenceIndicator = ({state}) => {
  let presence = state.presence
  let numStudentsOnline = presence && presence.students ? presence.students : 0
  let numStaffOnline = presence && presence.staff ? presence.staff : 0
  let color = numStaffOnline ? 'success' : 'warning'
  let pendingTickets = getTickets(state, 'pending');
  let assignedTickets = getTickets(state, 'assigned');
  let myTicket = getMyTicket(state)

  //Message displayed at front regarding # students and assistants online
  var studentMessage = numStudentsOnline + " students"
  var staffMessage =  numStaffOnline + " assistants"

  if (numStudentsOnline === 1) {
    studentMessage = studentMessage.slice(0, -1)
  }
  if (numStaffOnline === 1) {
    staffMessage = staffMessage.slice(0, -1)
  }

  let message = studentMessage + " and " + staffMessage + " currently online."

  // How many assistants are unoccupied
  var availableAssistants = numStaffOnline - assignedTickets.length

  // How many students need help, assuming all avaiable assistants are assigned
  var stillNeedHelp = Math.max(0, pendingTickets.length - availableAssistants)

  // Formatting
  var waitColor = "#646468"

  if (myTicket == undefined || myTicket.status != "pending") {
    //Calculate a generalized waittime, using the # people on queue + 1th position

    // PARAM 1: expected waittime time
    var avgHelpTime = getAvgWaitTime(state, state.waitTimes.length - 1)

    // catch if there actually are no assistants available
    if (numStaffOnline == 0) {
      var timeRange = "??"
    } else {
      // standard deviation for non-active ticket's gamma distribution
      var stdDev = getStdDev(state, state.waitTimes.length - 1)

      // PARAM 2: (75% conf interval by CLT, 1.15 is from zscore of Normal)
      var bound = 1.15 * stdDev/Math.sqrt(numStaffOnline)

      // interval bounds
      var estWaitTimeMin = Math.max(0, Math.floor(avgHelpTime - bound))
      var estWaitTimeMax = Math.ceil(avgHelpTime + bound)

      // colors for the time
      if (expectedWaitTotal <= 5) {
        waitColor ="#009900"
      } else if (expectedWaitTotal < 10) {
        waitColor ="#739900"
      } else if (expectedWaitTotal < 25) {
        waitColor ="#cc5200"
      } else {
        waitColor ="#ff0000"
      }

      // concatenate time range string
      var timeRange = estWaitTimeMin + " - " + estWaitTimeMax
      if (estWaitTimeMax > 120) {
        timeRange = "> 120"
      }
    }
  } else {

    //Get queue position of ticket
    var queue_position = ticketPositionIndex(state, ticket)

    // PARAM 1: expected waittime
    var avgWaitTime = getAvgWaitTime(state, queue_position)

    // catch if there actually are no assistants available
    if (numStaffOnline == 0) {
      var timeRange = "??"
    } else {

      // standard deviation
      var stdDev = getStdDev(state, queue_position)

      // PARAM 2: (75% conf interval by CLT, 1.15 is from zscore of Normal)
      var bound = 1.15 * stdDev/Math.sqrt(numStaffOnline)

      // interval bounds
      var estWaitTimeMin = Math.max(0, Math.floor(avgWaitTime - bound))
      var estWaitTimeMax = Math.ceil(avgWaitTime + bound)

      // colors for the time
      if (expectedWaitTotal <= 5) {
        waitColor ="#009900"
      } else if (expectedWaitTotal < 10) {
        waitColor ="#739900"
      } else if (expectedWaitTotal < 25) {
        waitColor ="#cc5200"
      } else {
        waitColor ="#ff0000"
      }

      // concatenate time range string
      var timeRange = estWaitTimeMin + " - " + estWaitTimeMax
      if (estWaitTimeMax > 120) {
        timeRange = "> 120"
      }
    }

  }

  // app.refresh();

  var welcomeMessage = state.config.welcome

  return (
    <div className="col-xs-12">

      <div className="alert alert-info alert-dismissable fade in" role="alert">
        <button type="button" className="close" aria-label="Close" data-dismiss="alert">
            <span aria-hidden="true">&times;</span>
        </button>
        <ReactMarkdown source={welcomeMessage} />
      </div>

      <div className={`alert alert-${color} alert-dismissable fade in`} role="alert">
        <button type="button" className="close" aria-label="Close" data-dismiss="alert">
            <span aria-hidden="true">&times;</span>
        </button>
        <h4>Estimated wait time: <font color={waitColor}><strong>{timeRange}</strong></font> minutes</h4>
        <h5>{ message }</h5>
        <MagicWordDisplay state={state} />
      </div>
    </div>
  );
}

