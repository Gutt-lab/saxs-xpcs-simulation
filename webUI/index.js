
async function thumbnailClicked() {
    let btn = document.getElementById('sample')
    return new Promise((resolve, reject) => {
        btn.addEventListener('click',async function(e) {
            const imgStr = await eel.generate_graph()();
            document.getElementById('plot2').src = 'data:image/png;base64,' + imgStr;
            resolve(true);
        });
      });
    const imgStr = await eel.generate_graph()();
    document.getElementById('plot').src = 'data:image/png;base64,' + imgStr;

//   eel.sample_clicked(id)(function(res){                       
//     // Update the div with a random number returned by python 
//     document.getElementById('ll').style.display = 'block'
//     document.querySelector("#title").innerHTML = res; 


//   }) 
}


async function test() {
    let btn = document.getElementById('xray')
    return new Promise((resolve, reject) => {
        btn.addEventListener('click',async function(e) {
            const imgStr = await eel.t()();
            if(imgStr)document.getElementById('plot').src = "img/frames_0.5_.gif";
            resolve(true);
        });
      });
}
thumbnailClicked()
test()