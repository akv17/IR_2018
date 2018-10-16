function activateElement(buttonId, elementId){
    let button = document.getElementById(buttonId);
    let el = document.getElementById(elementId);

    if (el.style.display == "none"){
        el.style.display = "";

        button.style.background = "#0099ff";
        button.style.color = "white";
    }

    else {
        el.style.display = "none";
        button.style.background = "";
        button.style.color = "";
    }
}