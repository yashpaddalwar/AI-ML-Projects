jQuery('#churnprediction').validate({
    rules:{
        seniorcitizen:{
            required:true
        },
        monthlycharges:{
            required:true,
            range: [18.25,120]
        },
        totalcharges:{
            required:true,
            range:[19,8684]
        },
        gender:{
            required:true
        },
        partner:{
            required:true
        },
        dependent:{
            required:true
        },
        gender:{
            required:true
        },
        phoneservice:{
            required:true
        },
        multiplelinesphoneservice:{
            required:true
        },
        internetservice:{
            required:true
        },
        onlinesecurity:{
            required:true
        },
        onlinebackup:{
            required:true
        },
        deviceprotection:{
            required:true
        },
        techsupport:{
            required:true
        },
        streamingtv:{
            required:true
        },
        streamingmovies:{
            required:true
        },
        contract:{
            required:true
        },
        paperlessbilling:{
            required:true
        },
        paymentmethod:{
            required:true
        },
        tenure:{
            required:true
        }
    },
    messages:{
        seniorcitizen:{
            required: "This field is required!"
        },
        monthlycharges:{
            required: "This field is required!",
            range: "Monthly charges ranges from 18.25 to 120"
        },
        totalcharges:{
            required: "This field is required!",
            range: "Total charges ranges from 19 to 8684"
        },
        gender:{
            required: "This field is required!"
        },
        partner:{
            required: "This field is required!"
        },
        dependent:{
            required: "This field is required!"
        },
        phoneservice:{
            required: "This field is required!"
        },
        multiplelinesphoneservice:{
            required: "This field is required!"
        },
        internetservice:{
            required: "This field is required!"
        },
        onlinesecurity:{
            required: "This field is required!"
        },
        onlinebackup:{
            required: "This field is required!"
        },
        deviceprotection:{
            required: "This field is required!"
        },
        techsupport:{
            required: "This field is required!"
        },
        streamingtv:{
            required: "This field is required!"
        },
        streamingmovies:{
            required: "This field is required!"
        },
        contract:{
            required: "This field is required!"
        },
        paperlessbilling:{
            required: "This field is required!"
        },
        paymentmethod:{
            required: "This field is required!"
        },
        tenure:{
            required: "This field is required!"
        }
    },
    submitHandler:function(form){
        form.submit();
    }
});