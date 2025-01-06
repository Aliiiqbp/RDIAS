import c2pa
import json

'''
    -inputs:
        1. private.key
        2. certificate.pem
        3. signature algorithm: eg. es256
        4. manifest.json
        5. source image
        6. destination folder
        7. image format
'''

print("C2PA version ==", c2pa.version())
prv_key = open("c2pa_sample/private_es256.key", "rb").read()
certs = open("c2pa_sample/certs_es256.pem", "rb").read()
sign_info = c2pa.SignerInfo("es256", certs, prv_key, "http://timestamp.digicert.com")

manifest_json = json.dumps({
    "alg": "es256",
    "private_key": "es256_private.key",
    "sign_cert": "es256_certs.pem",
    "ta_url": "http://timestamp.digicert.com",

    "claim_generator": "TestApp",
    "title": "Test DIV2K Images",
    "assertions": [
        {
            "label": "stds.schema-org.CreativeWork",
            "data": {
                "@context": "https://schema.org",
                "@type": "CreativeWork",
                "author": [
                    {
                        "@type": "Person",
                        "name": "Ali Ghorbanpour"
                    }
                ]
            }
        },
        {
            "label": "c2pa.actions",
            "data": {
                "actions": [
                    {
                        "action": "c2pa.opened"
                    }
                ],
                "metadata": {
                    "reviewRatings": [
                        {
                            "code": "c2pa.unknown",
                            "explanation": "Something untracked happened",
                            "value": 4
                        }
                    ]
                }
            }
        },
        {
            "label": "my.assertion",
            "data": {
                "any_tag": "whatever I want"
            }
        }
    ]
})


# for i in range(1, 21):
#     result = c2pa.sign_file("c2pa_sample/images/" + str(i) + ".png",
#                         "c2pa_sample/signed-images/" + str(i) + "-c2pa.png",
#                         manifest=manifest_json,
#                         signer_info=sign_info,
#                         data_dir="c2pa_sample/signed-images")

for i in range(1, 6):
    result = c2pa.sign_file("c2pa_sample/" + str(1) + ".png",
                        "c2pa_sample/" + str(1) + ".png",
                        manifest=manifest_json,
                        signer_info=sign_info,
                        data_dir="c2pa_sample/")


# json_store = c2pa.read_file("c2pa_sample/signed-images/1-c2pa.png", "c2pa_sample/signed-images/")
# print(json_store)
# with open('c2pa_sample/output/801.json', 'w') as f:
#     json.dump(json_store, f)

# d32083ce-1045-4626-ab46-d051f05bf4bf