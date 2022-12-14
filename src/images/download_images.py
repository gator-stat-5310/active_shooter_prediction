from bs4 import *
import requests
import os


# CREATE FOLDER
def folder_create(images):
    try:
        #folder_name = input("Enter Folder Name:- ")
        folder_name = r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\presentation\SubMachineGun'
        # folder creation
        os.makedirs(folder_name,exist_ok=True)

    # if folder exists with that name, ask another name
    except:
        print("Folder Exist with that name!")
        folder_create()

    # image downloading start
    download_images(images, folder_name)


# DOWNLOAD ALL IMAGES FROM THAT URL
def download_images(images, folder_name):
    # initial count is zero
    count = 0

    # print total images found in URL
    print(f"Total {len(images)} Image Found!")

    # checking if images is not zero
    if len(images) != 0:
        for i, image in enumerate(images):
            # From image tag ,Fetch image Source URL

            # 1.data-srcset
            # 2.data-src
            # 3.data-fallback-src
            # 4.src

            # Here we will use exception handling

            # first we will search for "data-srcset" in img tag
            try:
                # In image tag ,searching for "data-srcset"
                image_link = image["data-srcset"]

            # then we will search for "data-src" in img
            # tag and so on..
            except:
                try:
                    # In image tag ,searching for "data-src"
                    image_link = image["data-src"]
                except:
                    try:
                        # In image tag ,searching for "data-fallback-src"
                        image_link = image["data-fallback-src"]
                    except:
                        try:
                            # In image tag ,searching for "src"
                            image_link = image["src"]

                        # if no Source URL found
                        except:
                            pass

            # After getting Image Source URL
            # We will try to get the content of image
            try:
                https_image_link=f'https://www.imfdb.org/{image_link}'
                # https_image_link= 'https://www.imfdb.org/wiki/File:Walther-PP-Post-War.jpg'
                r = requests.get(https_image_link).content
                try:
                    with open(f"{folder_name}/images{i + 1}.jpg", "wb+") as f:
                        f.write(r)

                    count += 1
                except Exception as ex:
                    print(str(ex))
            #
            #     try:
            #
            #         # possibility of decode
            #         r = str(r, 'utf-8')
            #
            #     except UnicodeDecodeError:
            #
            #         # After checking above condition, Image Download start
            #         with open(f"{folder_name}/images{i + 1}.jpg", "wb+") as f:
            #             f.write(r)
            #
            #         # counting number of image downloaded
            #         count += 1
            except Exception as ex:
                print(str(ex))

        # There might be possible, that all
        # images not download
        # if all images download
        if count == len(images):
            print("All Images Downloaded!")

        # if all images not download
        else:
            print(f"Total {count} Images Downloaded Out of {len(images)}")


# MAIN FUNCTION START
def main(url):
    # content of URL
    r = requests.get(url)

    # Parse HTML Code
    soup = BeautifulSoup(r.text, 'html.parser')

    # find all images in URL
    images = soup.findAll('img')

    # Call folder create function
    folder_create(images)


# take url
#url = input("Enter URL:- ")
pistol_url = "https://www.imfdb.org/wiki/Category:Pistol"
shot_guns_url = "https://www.imfdb.org/wiki/Category:Shotgun"
assault_rifles_url = "https://www.imfdb.org/wiki/Category:Assault_Rifle"
machine_gun_url = "https://www.imfdb.org/wiki/Category:Machine_Gun"
revolver_url = 'https://www.imfdb.org/wiki/Category:Revolver'
submachine_gun_url = "https://www.imfdb.org/wiki/Category:Submachine_Gun"
# CALL MAIN FUNCTION
main(submachine_gun_url)
