
// jQuery.validator.setDefaults({
//     debug: true,
//     success: "valid"
// });
// $(function(){
// $("form[name='admissionchecker']").validate({
//     $( "#admissionchecker" ).validate({
//          rules:{
//              gre:{
//                  required=true,
//                  range: [0, 340]
//              },
//              toefl:{
//                  required:true,
//                  range: [0, 120]
//              },
//              university:{
//                 required:true,
//                 range: [0, 5]
//              },
//              sop:{
//                 required:true,
//                 range: [0, 5]
//              },
//              lor:{
//                 required:true,
//                 range: [0, 5]
//              },
//              cgpa:{
//                 required:true,
//                 range: [0, 9.92]
//              },
//              research:"required"
//          },

//         messages:{
//             gre:{required:"Please fill this field",minLength:"GRE Score should be between 0 and 340"},
//             toefl:{required:"Please fill this field",minLength:"TOEFL Score should be between 0 and 120"},
//             university:{required:"Please fill this field",minLength:"University Rating should be between 0 and 5"},
//             sop:{required:"Please fill this field",minLength:"SOP Score should be between 0 and 5"},
//             lor:{required:"Please fill this field",minLength:"LOR Score should be between 0 and 5"},
//             cgpa:{required:"Please fill this field",minLength:"CGPA should be below 10"}
//         },
//         submitHandler: function(form){
//             form.submit()
//         }
// });
// });

jQuery('#admissionchecker').validate({
	rules:{
			age:{
				required:true,
				range: [18, 100]
			},
			sex:{
			    required:true,
			},
			bmi:{
			   required:true,
			   range: [16, 53.5]
			},
			children:{
			   required:true,
			   range: [0, 14]
			},
			smoker:{
			   required:true
			},
			region:{
			   required:true
			}
	},
    messages:{
		age:{
			required:"Please fill this field!",
			range:"Age should be greater than or equal to 18!"
		},
		sex:{
			required:"Please fill this field!"
		},
        bmi:{
			required:"Please fill this field!",
			range:"BMI ranges between 16 and 53.5"
		},
        children:{
			required:"Please fill this field!",
			range:"Please Enter a Valid Number!"
		},
        smoker:{
			required:"Please fill this field!"
		},
        region:{
			required:"Please fill this field!"
		}
	},
	submitHandler:function(form){
		form.submit();
	}
});