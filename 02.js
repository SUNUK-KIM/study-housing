$.ajax({
    url: 'test-target.csv',
    dataType: 'text',
}).done(successFunction);

function successFunction(data) {
    var allRows = data.split(/\r?\n|\r/);
    var table = '<table>';
    for (var singleRow = 0; singleRow < allRows.length; singleRow++) {
        if (singleRow === 0) {
            table += '<thead>';
            table += '<tr>';
        } else {
            table += '<tr>';
        }
        var rowCells = allRows[singleRow].split(',');
        for (var rowCell = 0; rowCell < rowCells.length; rowCell++) {
            if (singleRow === 0) {
                table += '<th>';
                table += rowCells[rowCell];
                table += '</th>';
            } else {
                table += '<td>';
                table += rowCells[rowCell];
                table += '</td>';
            }
        }
        if (singleRow === 0) {
            table += '</tr>';
            table += '</thead>';
            table += '<tbody>';
        } else {
            table += '</tr>';
        }
    }
    table += '</tbody>';
    table += '</table>';
    $('body').append(table);
}
table {
    margin: 0 auto;
    text - align: center;
    border - collapse: collapse;
    border: 1px solid #d4d4d4;
}

tr: nth - child(even) {
    background: #d4d4d4;
}

th, td {
    padding: 10px 30px;
}

th {
    border - bottom: 1px solid #d4d4d4;
}     