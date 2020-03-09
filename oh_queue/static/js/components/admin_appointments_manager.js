function AdminAppointmentsManager({ state }) {
    const [sheetUrl, setSheetUrl] = React.useState("");
    const [sheetName, setSheetName] = React.useState("");

    const handleSheetUrlChange = (e) => {
        setSheetUrl(e.target.value);
    };
    const handleSheetNameChange = (e) => {
        setSheetName(e.target.value);
    };

    const submit = () => {
        app.makeRequest("upload_appointments", {
            sheetUrl, sheetName,
        }, true);
    };

    return (
        <React.Fragment>
            <AdminOptionsManager>
                <tr>
                    <td>Should students be able to make appointments?</td>
                    <td className="col-md-1">
                        <ConfigLinkedToggle
                            config={state.config}
                            configKey="appointments_open"
                            offText="No"
                            onText="Yes"
                        />
                    </td>
                </tr>
            </AdminOptionsManager>
            <form>
                <div className="input-group appointment-input">
                    <input id="url-selector" type="text" className="form-control" placeholder="Link to a spreadsheet containing appointments" required value={sheetUrl} onChange={handleSheetUrlChange} />
                    <input id="sheet-selector" className="form-control form-right" type="text" name="question" title="Sheet name" placeholder="Sheet name" required value={sheetName} onChange={handleSheetNameChange}/>
                      <span className="input-group-btn">
                        <button className="btn btn-default" type="button" onClick={submit}>Update</button>
                      </span>
                </div>
            </form>
        </React.Fragment>
    );
}
