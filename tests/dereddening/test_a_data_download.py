"""
Test for the data download functionality in the dereddening module.
"""

def test_can_download_data():
    from quasar_utils.dereddening.download_data import main
    main(force=True)