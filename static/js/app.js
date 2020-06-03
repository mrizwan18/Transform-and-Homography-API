function create_user(event) {

    //Function to fetch data of the user to be created from form and then make an ajax call of type POST to api

    event.preventDefault();
    let uId = document.getElementById("userID").value;
    let u_email = document.getElementById("userEmail").value;
    let f_name = document.getElementById("userFirstName").value;
    let l_name = document.getElementById("userLastName").value;
    let u_avatar = document.getElementById("userAvatar").value;
    let user = { id: uId, email: u_email, fname: f_name, lname: l_name, avatar: u_avatar }
    if (validateForm(user)) {
        user = JSON.parse(JSON.stringify(user));
        $.ajax({
            url: '/create', //endpoint of API
            contentType: "application/json;charset=utf-8",
            data: JSON.stringify({ user }),
            dataType: "json",
            type: 'POST',
            success: function (response) {
            },
            error: function (error) {
                if (error.status != 200)
                    document.getElementById("status").innerHTML = "Some error occurred ";
                else
                    document.getElementById("status").innerHTML = "User successfully created";
            }
        });
    }
    else
        document.getElementById("status").innerHTML = "No input feild can be empty ";

}
function delete_user(event) {

    //Function to fetch id of the user to be deleted from form and then make an ajax call of type DELETE to api

    event.preventDefault();
    let uid = document.getElementById("userID").value;
    let url = '/delete/' + uid
    $.ajax({
        url: url,
        type: 'DELETE',
        success: function (result) {
            document.getElementById("status").innerHTML = "User successfully deleted";
        },
        error: function (error) {
            document.getElementById("status").innerHTML = "Some error occurred ";
        }
    });
};
function update_user(event) {

    //Function to fetch data of the user to be updated from form and then make an ajax call of type PUT to api

    event.preventDefault();
    let uId = document.getElementById("userID").value;
    let u_email = document.getElementById("userEmail").value;
    let f_name = document.getElementById("userFirstName").value;
    let l_name = document.getElementById("userLastName").value;
    let u_avatar = document.getElementById("userAvatar").src;
    let user = { id: uId, email: u_email, fname: f_name, lname: l_name, avatar: u_avatar }
    if (validateForm(user)) {
        user = JSON.parse(JSON.stringify(user));
        let url = '/update/' + uId
        $.ajax({
            url: url,
            type: 'PUT',
            contentType: "application/json;charset=utf-8",
            data: JSON.stringify({ user }),
            dataType: "json",
            success: function (result) {
            },
            error: function (error) {
                if (error.status != 200)
                    document.getElementById("status").innerHTML = "Some error occurred ";
                else
                    document.getElementById("status").innerHTML = "User successfully updated";
            }
        });
    }
    else
        document.getElementById("status").innerHTML = "No Input Field can be empty";
}
function get_user(event) {

    //Function to fetch if of the user to be displayed from form and then make an ajax call of type GET to api

    event.preventDefault();
    let uId = event.target.getAttribute('data-arg1');;
    let url = '/users/' + uId
    $.ajax({
        url: url,
        type: 'GET',
        dataType: 'html',
        success: function (result) {
            $("html").html(result); //returned HTML response is then rendered on current page
        },
        error: function (error) {
            if (error.status != 200)
                document.getElementById("status").innerHTML = "Some error occurred ";
            else
                document.getElementById("status").innerHTML = "User successfully updated";
        }
    });
}

function validateForm(user) {

    //Function to check if all the fields are filled before making any API call

    return (user.id && user.email && user.fname && user.lname && user.avatar)
}